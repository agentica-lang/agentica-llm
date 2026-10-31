## Bindings for the part of llama.cpp (llama.h, release b12012) that a server needs, with a small wrapper.
##
## The struct layouts are copied from llama.h of the same tag: a wrong field list is undefined behavior, so
## `scripts/fetch-headers.sh` gets the header to compare with. Enums are `C.Int32`.

import Numbers from "numbers"

BYTES_PER_TOKEN = 4
POINTER_BYTES = 8
CHAT_MESSAGE_BYTES = 16
MAX_UBATCH = 512

## The size of the scratch buffer `piece` needs.
export PIECE_BYTES = 256

extern "llama"
  export resource Model, free: llama_model_free
  export resource Context, free: llama_free
  export resource Sampler, free: llama_sampler_free

  struct ModelParams
    devices: C.Pointer
    tensor_buft_overrides: C.Pointer
    n_gpu_layers: C.Int32
    split_mode: C.Int32
    load_mode: C.Int32
    lazy_mode: C.Int32
    main_gpu: C.Int32
    tensor_split: C.Pointer
    progress_callback: C.Pointer
    progress_callback_user_data: C.Pointer
    kv_overrides: C.Pointer
    vocab_only: C.Bool
    check_tensors: C.Bool
    use_extra_bufts: C.Bool
    no_host: C.Bool
    no_alloc: C.Bool
    load_mtp: C.Bool
  end

  struct ContextParams
    n_ctx: C.UInt32
    n_batch: C.UInt32
    n_ubatch: C.UInt32
    n_seq_max: C.UInt32
    n_rs_seq: C.UInt32
    n_outputs_max: C.UInt32
    n_outputs_max_per_seq: C.UInt32
    n_threads: C.Int32
    n_threads_batch: C.Int32
    ctx_type: C.Int32
    rope_scaling_type: C.Int32
    pooling_type: C.Int32
    attention_type: C.Int32
    flash_attn_type: C.Int32
    rope_freq_base: C.Float
    rope_freq_scale: C.Float
    yarn_ext_factor: C.Float
    yarn_attn_factor: C.Float
    yarn_beta_fast: C.Float
    yarn_beta_slow: C.Float
    yarn_orig_ctx: C.UInt32
    defrag_thold: C.Float
    cb_eval: C.Pointer
    cb_eval_user_data: C.Pointer
    type_k: C.Int32
    type_v: C.Int32
    moe_cache_size: C.Int64
    abort_callback: C.Pointer
    abort_callback_data: C.Pointer
    embeddings: C.Bool
    offload_kqv: C.Bool
    no_perf: C.Bool
    op_offload: C.Bool
    swa_full: C.Bool
    kv_unified: C.Bool
    samplers: C.Pointer
    n_samplers: C.Int64
    ctx_other: C.Pointer
  end

  struct Batch
    n_tokens: C.Int32
    token: C.Pointer
    embd: C.Pointer
    pos: C.Pointer
    n_seq_id: C.Pointer
    seq_id: C.Pointer
    logits: C.Pointer
  end

  struct SamplerChainParams
    no_perf: C.Bool
  end

  def llama_backend_init() -> Void
  def llama_model_default_params() -> ModelParams
  def llama_context_default_params() -> ContextParams
  def llama_sampler_chain_default_params() -> SamplerChainParams

  ## Reads the whole file: seconds, so it runs on the compute pool.
  compute def llama_model_load_from_file(path: C.String, params: ModelParams) -> Model
  def llama_init_from_model(model: Model, params: ContextParams) -> Context

  def llama_model_get_vocab(model: Model) -> C.Pointer
  def llama_model_chat_template(model: Model, name: C.Pointer) -> C.Pointer
  def llama_vocab_n_tokens(vocab: C.Pointer) -> C.Int32
  def llama_vocab_is_eog(vocab: C.Pointer, token: C.Int32) -> C.Bool
  def llama_tokenize(vocab: C.Pointer, text: C.Pointer, text_len: C.Int32, tokens: C.Pointer, n_tokens_max: C.Int32,
                     add_special: C.Bool, parse_special: C.Bool) -> C.Int32
  def llama_token_to_piece(vocab: C.Pointer, token: C.Int32, buf: C.Pointer, length: C.Int32, lstrip: C.Int32,
                           special: C.Bool) -> C.Int32
  def llama_chat_apply_template(tmpl: C.Pointer, chat: C.Pointer, n_msg: C.Int64, add_ass: C.Bool, buf: C.Pointer,
                                length: C.Int32) -> C.Int32

  def llama_batch_init(n_tokens: C.Int32, embd: C.Int32, n_seq_max: C.Int32) -> Batch
  def llama_batch_free(batch: Batch) -> Void

  ## One step over the whole batch: the long call of the engine.
  compute def llama_decode(ctx: Context, batch: Batch) -> C.Int32

  def llama_get_memory(ctx: Context) -> C.Pointer
  def llama_memory_seq_rm(mem: C.Pointer, seq_id: C.Int32, p0: C.Int32, p1: C.Int32) -> C.Bool

  def llama_sampler_chain_init(params: SamplerChainParams) -> Sampler
  def llama_sampler_chain_add(chain: Sampler, smpl: C.Pointer) -> Void
  def llama_sampler_init_greedy() -> C.Pointer
  def llama_sampler_init_dist(seed: C.UInt32) -> C.Pointer
  def llama_sampler_init_top_k(k: C.Int32) -> C.Pointer
  def llama_sampler_init_temp(t: C.Float) -> C.Pointer
  def llama_sampler_sample(smpl: Sampler, ctx: Context, idx: C.Int32) -> C.Int32
end

## One message of a chat, as the OpenAI API names it.
export struct ChatMessage
  role: String
  content: String
end

## One token of a batch: the slot is its sequence id.
export struct BatchItem
  token: Int
  pos: Int
  slot: Int
  wants_logits: Bool
end

## The batch arrays are allocated once for `capacity` tokens and rewritten at each step.
export struct BatchBuffer
  batch: Batch
  capacity: Int
end

## Loads the CPU backends that sit in `backend_dir` (the release builds them as separate libraries) and starts
## llama.cpp.
export def init(backend_dir: String) -> Void
  ggml_backend_load_all_from_path(backend_dir)
  llama_backend_init()
end

export def load_model(path: String) -> Result[Model]
  model = llama_model_load_from_file(path, llama_model_default_params().with(n_gpu_layers: 0))
  case model
    ok(m) -> return ok(m)
    error(_) -> return error("cannot load the model at #{path}")
  end
end

## A context with `n_seq` sequences that share `n_ctx` tokens of KV cache, so each holds `n_ctx / n_seq`.
export def new_context(model: Model, n_ctx: Int, n_batch: Int, n_seq: Int, threads: Int) -> Result[Context]
  params = llama_context_default_params().with(
    n_ctx: n_ctx,
    n_batch: n_batch,
    n_ubatch: Numbers.smaller(n_batch, MAX_UBATCH),
    n_seq_max: n_seq,
    n_threads: threads,
    n_threads_batch: threads,
    no_perf: true,
    kv_unified: false,
  )
  case llama_init_from_model(model, params)
    ok(ctx) -> return ok(ctx)
    error(_) -> return error("cannot create the context")
  end
end

## The text of a chat as the model's own template makes it, ready to tokenize.
export def chat_prompt(model: Model, messages: List[ChatMessage]) -> Result[String]
  count = messages.size
  total = messages.reduce(0) do |sum, m| sum + m.role.bytesize + m.content.bytesize end
  # One message holds two C strings; their buffers live until the call returns.
  strings = messages.map do |m| [Native.from_string(m.role), Native.from_string(m.content)] end
  chat = Native.memory_pointer(count * CHAT_MESSAGE_BYTES)
  for i in 0...count
    Native.put_pointer(chat, i * CHAT_MESSAGE_BYTES, strings[i][0])
    Native.put_pointer(chat, i * CHAT_MESSAGE_BYTES + POINTER_BYTES, strings[i][1])
  end

  tmpl = llama_model_chat_template(model, Native.null())
  size = total * 2 + 256 + count * 32
  buf = Native.memory_pointer(size)
  written = llama_chat_apply_template(tmpl, chat, count, true, buf, size)
  text = ""
  if written > size
    # The template grew past the guess: ask again with the exact size.
    Native.free(buf)
    buf = Native.memory_pointer(written)
    written = llama_chat_apply_template(tmpl, chat, count, true, buf, written)
  end

  if written >= 0
    text = Native.get_string(buf, 0, written)
  end

  Native.free(buf)
  Native.free(chat)
  for pair in strings
    Native.free(pair[0])
    Native.free(pair[1])
  end

  if written < 0
    return error("the chat template does not apply to these messages")
  end

  return ok(text)
end

## The token ids of `text`, special tokens (such as `<|im_start|>`) included.
export def tokenize(model: Model, text: String) -> Result[List[Int]]
  vocab = llama_model_get_vocab(model)
  size = text.bytesize + 16
  input = Native.from_string(text)
  out = Native.memory_pointer(size * BYTES_PER_TOKEN)
  count = llama_tokenize(vocab, input, text.bytesize, out, size, false, true)
  tokens: List[Int] = []
  if count >= 0
    tokens = Native.get_array_of_int32(out, 0, count)
  end

  Native.free(out)
  Native.free(input)
  if count < 0
    return error("cannot tokenize the prompt")
  end

  return ok(tokens)
end

export def eog?(model: Model, token: Int) -> Bool
  return llama_vocab_is_eog(llama_model_get_vocab(model), token)
end

## The bytes of one token. A character can be split between two tokens, so the caller joins bytes, not strings.
## `scratch` is a buffer of at least `PIECE_BYTES` that the caller owns.
export def piece(model: Model, token: Int, scratch: C.Pointer) -> Bytes
  n = llama_token_to_piece(llama_model_get_vocab(model), token, scratch, PIECE_BYTES, 0, false)
  if n <= 0
    return "".bytes
  end

  return Native.get_bytes(scratch, 0, n)
end

## A sampler chain that runs in C: greedy at temperature 0, else temperature and a seeded draw.
export def new_sampler(temperature: Float, seed: Int) -> Result[Sampler]
  chain = llama_sampler_chain_init(llama_sampler_chain_default_params())
  case chain
    ok(sampler) ->
      if temperature <= 0.0
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy())
      else
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature))
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed))
      end

      return ok(sampler)
    error(_) -> return error("cannot create a sampler")
  end
end

## The next token of the sequence whose logits are at row `row` of the last decoded batch.
export def sample(sampler: Sampler, ctx: Context, row: Int) -> Int
  return llama_sampler_sample(sampler, ctx, row)
end

export def new_batch(capacity: Int) -> BatchBuffer
  # One sequence id per token, so a batch row never belongs to two sequences.
  return BatchBuffer(batch: llama_batch_init(capacity, 0, 1), capacity: capacity)
end

## Writes `items` into the batch and returns the batch to pass to `decode`.
export def fill(buffer: BatchBuffer, items: List[BatchItem]) -> Batch
  b = buffer.batch
  n = items.size
  Native.put_array_of_int32(b.token, 0, items.map do |i| i.token end)
  Native.put_array_of_int32(b.pos, 0, items.map do |i| i.pos end)
  Native.put_array_of_int32(b.n_seq_id, 0, items.map do |_| 1 end)
  Native.put_bytes(b.logits, 0, logit_flags(items))
  for k in 0...n
    cell = Native.get_pointer(b.seq_id, k * POINTER_BYTES)
    Native.put_int32(cell, 0, items[k].slot)
  end

  return b.with(n_tokens: n)
end

## Runs the model over the batch. 0 is success.
export def decode(ctx: Context, batch: Batch) -> Int
  return llama_decode(ctx, batch)
end

## Forgets the KV cache of one sequence, so its slot can take another request.
export def clear_slot(ctx: Context, slot: Int) -> Void
  _ = llama_memory_seq_rm(llama_get_memory(ctx), slot, -1, -1)
end

def logit_flags(items: List[BatchItem]) -> Bytes
  flags = items.map do |i| flag(i.wants_logits) end
  return Bytes.from_list(flags).unwrap!
end

def flag(on: Bool) -> Int
  if on
    return 1
  end

  return 0
end

extern "ggml"
  def ggml_backend_load_all_from_path(dir_path: C.String) -> Void
end
