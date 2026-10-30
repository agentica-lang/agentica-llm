## Generates one answer on one sequence, to check the bindings: `agentica smoke.agc`.

import Llama from "src/llama"

def generate(model: Llama.Model, ctx: Llama.Context, prompt: List[Int], limit: Int) -> Result[String]
  sampler = try Llama.new_sampler(0.0, 1)
  buffer = Llama.new_batch(512)
  scratch = Native.memory_pointer(Llama.PIECE_BYTES)
  items: List[Llama.BatchItem] = []
  last = prompt.size - 1
  for i in 0...prompt.size
    items = items.push(Llama.BatchItem(token: prompt[i], pos: i, slot: 0, wants_logits: i == last))
  end
  parts: List[Bytes] = []
  pos = prompt.size
  for _ in 0...limit
    rc = Llama.decode(ctx, Llama.fill(buffer, items))
    if rc != 0
      return error("decode failed: #{rc}")
    end
    token = Llama.sample(sampler, ctx, items.size - 1)
    if Llama.eog?(model, token)
      break
    end
    parts = parts.push(Llama.piece(model, token, scratch))
    items = [Llama.BatchItem(token: token, pos: pos, slot: 0, wants_logits: true)]
    pos += 1
  end
  return String.from_bytes(Bytes.concat(parts))
end

def run(model_path: String) -> Result[Void]
  Llama.init("native/linux")
  model = try Llama.load_model(model_path)
  ctx = try Llama.new_context(model, 2048, 512, 1, 4)
  text = try Llama.chat_prompt(model, [
    Llama.ChatMessage(role: "system", content: "You are a helpful assistant."),
    Llama.ChatMessage(role: "user", content: "Say hello in five words."),
  ])
  puts(text)
  tokens = try Llama.tokenize(model, text)
  puts("prompt tokens: #{tokens.size}")
  answer = try generate(model, ctx, tokens, 32)
  puts(answer)
  return ok()
end

def main() -> Void
  case run("deps/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    ok(_) -> puts("done")
    error(message) -> puts("error: #{message}")
  end
end
