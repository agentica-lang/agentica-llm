## The engine: one task owns the llama.cpp context and does continuous batching.
##
## Requests arrive on a channel. Each step puts the next token of every active sequence in one batch (a new
## sequence brings its whole prompt), decodes it once, samples each sequence in C and answers the sequences that
## finished. A free slot takes the next waiting request at once, so a long answer never holds the others back.

import Llama from "llama"

IDLE_WAIT_MS = 100
REPLY_TIMEOUT_MS = 5000

## A request already tokenized by the route; the answer goes back on `reply`.
export struct GenRequest
  prompt: List[Int]
  max_tokens: Int
  temperature: Float
  seed: Int
  reply: Channel[GenResult]
end

export struct GenResult
  text: String
  prompt_tokens: Int
  completion_tokens: Int
  finish_reason: String
  failure: String
end

export struct EngineConfig
  n_ctx: Int
  n_batch: Int
  n_parallel: Int
  threads: Int
end

# What the engine needs at each step; the context is owned by the engine task.
struct Engine
  model: Llama.Model
  ctx: Llama.Context
  buffer: Llama.BatchBuffer
  scratch: C.Pointer
  config: EngineConfig
end

# A running request. `pending` holds the tokens not decoded yet: the prompt first, then the last sampled token.
struct Seq
  slot: Int
  request: GenRequest
  sampler: Llama.Sampler
  limit: Int
  pending: List[Int]
  pos: Int
  parts: List[Bytes]
  generated: Int
end

# The share of one step that belongs to a sequence: `fed` tokens, the last of them at batch row `row` (-1 when
# the prompt is not finished and no token is sampled yet).
struct Feed
  seq: Seq
  fed: Int
  row: Int
end

## Runs until the process ends; returns an error only when the context cannot be created.
export def run(model: Llama.Model, config: EngineConfig, requests: Channel[GenRequest]) -> Result[Void]
  ctx = try Llama.new_context(model, config.n_ctx, config.n_batch, config.n_parallel, config.threads)
  engine = Engine(
    model: model,
    ctx: ctx,
    buffer: Llama.new_batch(config.n_batch),
    scratch: Native.memory_pointer(Llama.PIECE_BYTES),
    config: config,
  )
  seqs: List[Seq] = []
  while true
    seqs = admit(engine, seqs, requests)
    if not seqs.empty?
      seqs = step(engine, seqs)
    end
  end
  return ok()
end

# Takes waiting requests while a slot is free. Idle, it waits a little for the first one; busy, it only looks.
def admit(engine: Engine, seqs: List[Seq], requests: Channel[GenRequest]) -> List[Seq]
  active = seqs
  while active.size < engine.config.n_parallel
    wait = 0
    if active.empty?
      wait = IDLE_WAIT_MS
    end
    case requests.recv(wait)
      ok(request) ->
        case start(engine, active, request)
          ok(seq) -> active = active.push(seq)
          error(_) -> next
        end
      error(_) -> break
    end
  end
  return active
end

# Gives the request the lowest free slot. A request that cannot run is answered here and dropped.
def start(engine: Engine, active: List[Seq], request: GenRequest) -> Result[Seq]
  per_seq = engine.config.n_ctx / engine.config.n_parallel
  if request.prompt.empty? or request.prompt.size >= per_seq
    refuse(request, "the prompt does not fit in a slot of #{per_seq} tokens")
    return error("refused")
  end
  case Llama.new_sampler(request.temperature, request.seed)
    ok(sampler) ->
      return ok(Seq(
        slot: free_slot(active),
        request: request,
        sampler: sampler,
        limit: smaller(request.max_tokens, per_seq - request.prompt.size),
        pending: request.prompt,
        pos: 0,
        parts: [],
        generated: 0,
      ))
    error(message) ->
      refuse(request, message)
      return error("refused")
  end
end

def free_slot(active: List[Seq]) -> Int
  taken = active.map do |s| s.slot end
  slot = 0
  while taken_in?(taken, slot)
    slot += 1
  end
  return slot
end

def taken_in?(taken: List[Int], slot: Int) -> Bool
  return taken.any? do |t| t == slot end
end

# One batch: decode it, then sample and settle each sequence.
def step(engine: Engine, seqs: List[Seq]) -> List[Seq]
  feeds = plan(engine.config.n_batch, seqs)
  items = feeds.flat_map do |f| items_of(f) end
  rc = Llama.decode(engine.ctx, Llama.fill(engine.buffer, items))
  if rc != 0
    seqs.each do |s| fail_seq(engine, s, "decode failed with code #{rc}") end
    return []
  end
  start_list: List[Seq] = []
  return feeds.reduce(start_list) do |kept, f| settle(engine, kept, f) end
end

# Gives each sequence its pending tokens, in order, while the batch has room.
def plan(budget: Int, seqs: List[Seq]) -> List[Feed]
  feeds: List[Feed] = []
  used = 0
  for seq in seqs
    fed = smaller(seq.pending.size, budget - used)
    if fed > 0
      used += fed
      row = -1
      if fed == seq.pending.size
        row = used - 1
      end
      feeds = feeds.push(Feed(seq: seq, fed: fed, row: row))
    end
  end
  return feeds
end

def items_of(feed: Feed) -> List[Llama.BatchItem]
  seq = feed.seq
  tokens = seq.pending.take(feed.fed)
  items: List[Llama.BatchItem] = []
  for i in 0...feed.fed
    items = items.push(Llama.BatchItem(
      token: tokens[i],
      pos: seq.pos + i,
      slot: seq.slot,
      wants_logits: feed.row >= 0 and i == feed.fed - 1,
    ))
  end
  return items
end

# Adds the sequence to `kept` unless it finished.
def settle(engine: Engine, kept: List[Seq], feed: Feed) -> List[Seq]
  seq = feed.seq.with(pending: feed.seq.pending.drop(feed.fed), pos: feed.seq.pos + feed.fed)
  if feed.row < 0
    return kept.push(seq)
  end
  token = Llama.sample(seq.sampler, engine.ctx, feed.row)
  if Llama.eog?(engine.model, token)
    finish(engine, seq, "stop")
    return kept
  end
  grown = seq.with(
    pending: [token],
    parts: seq.parts.push(Llama.piece(engine.model, token, engine.scratch)),
    generated: seq.generated + 1,
  )
  if grown.generated >= grown.limit
    finish(engine, grown, "length")
    return kept
  end
  return kept.push(grown)
end

def finish(engine: Engine, seq: Seq, reason: String) -> Void
  # Bytes that are not UTF-8 (a character cut by the limit) fail the conversion: keep the readable part.
  text = String.from_bytes(Bytes.concat(seq.parts)).unwrap_or("")
  release(engine, seq)
  answer = GenResult(
    text: text,
    prompt_tokens: seq.request.prompt.size,
    completion_tokens: seq.generated,
    finish_reason: reason,
    failure: "",
  )
  _ = seq.request.reply.send(answer, REPLY_TIMEOUT_MS)
end

def fail_seq(engine: Engine, seq: Seq, message: String) -> Void
  release(engine, seq)
  refuse(seq.request, message)
end

def release(engine: Engine, seq: Seq) -> Void
  Llama.clear_slot(engine.ctx, seq.slot)
  Native.close(seq.sampler)
end

def refuse(request: GenRequest, message: String) -> Void
  answer = GenResult(text: "", prompt_tokens: 0, completion_tokens: 0, finish_reason: "error", failure: message)
  _ = request.reply.send(answer, REPLY_TIMEOUT_MS)
end

def smaller(a: Int, b: Int) -> Int
  if a < b
    return a
  end
  return b
end
