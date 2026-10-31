# Architecture

The [Bond](https://github.com/agentica-lang/bond) web framework in front of llama.cpp, called through the Agentica FFI
(no C code). Written for Agentica 0.19.0-beta.2 and Bond 0.3.0 (the tag pinned in `agentica.toml`).

## How it works

- [src/llama.agc](../src/llama.agc): bindings for the part of `llama.h` a server needs (backend init, model load, context, vocab,
  tokenize, chat template, batch, decode, sampler chain, token to piece, free). `llama_decode` and the model load are
  `compute def`. The `llama_*_params` structs and `llama_batch` are FFI structs passed by value.
- [src/engine.agc](../src/engine.agc): one task owns the context and does continuous batching. Requests arrive on a `Channel`; each step
  puts the next token of every active sequence (or the whole prompt of a new one) into one batch, up to
  `--parallel` sequences (each its own `seq_id`, its own `n_ctx / parallel` share of the KV cache), decodes once,
  samples every sequence with llama.cpp's sampler chain in C, and answers on a per-request channel. A freed slot takes the
  next waiting request in the next step.
- [src/api.agc](../src/api.agc), [server.agc](../server.agc): Bond routes `POST /v1/chat/completions` (OpenAI shape:
  `messages`, `max_tokens`, `temperature`, `seed`, `stream`, `stream_options.include_usage`) and `GET /health`.
  A plain request is answered with `Bond.json_as(conn, 200, JSON.codec(ChatResponse), response)`.

## The streaming path

`"stream": true` answers `text/event-stream` with `Bond.sse`, in the shape `llama-server` and OpenAI emit: a chunk with
`delta.role`, a chunk with `delta.content` per piece of text, a chunk with `finish_reason` and an empty `delta`, a usage
chunk when `stream_options.include_usage` is on, then `data: [DONE]`.

```text
client <-SSE- Bond (std/http) <- events: Channel[Bytes] <- pump task <- reply: Channel[GenResult] <- engine task
```

- **The engine sends each token's text as soon as it is sampled**, on the request's own `reply` channel, as a
  `GenResult` with an empty `finish_reason`; the last one carries the finish reason and the counts. It sends with a
  timeout of 0: it never waits for a reader. The route sizes the channel for the whole answer
  (`min(max_tokens, n_ctx) + 2`), so a reader that keeps up never sees it full.
- **UTF-8 stays whole.** A character can be split between tokens (an emoji often is). The engine keeps the bytes of an
  unfinished character in the sequence (`held`) and sends only whole characters, so every event is valid text. Bytes
  left over when the answer ends are dropped.
- **The route waits for the first piece** before it returns `Bond.sse`, so a request the engine refuses (a prompt that
  does not fit a slot) is still a `400` and the time to first token is real. It then starts the `pump` task and returns:
  the `Conn` is gone once the handler returns, so the task gets the channels and the ids, not the `Conn`.
- **The pump** turns pieces into chunk events, sends them to Bond's `events` channel with a timeout, and closes it at the
  end, which ends the HTTP body. It runs with `Task.spawn_owning`, so it owns `reply` and `events`: if it crashes, both
  close and the response ends instead of hanging. The channel holds the whole answer, so the engine's send fails only
  when the pump closed `reply`.
- **A client that leaves.** Bond closes `events` when the connection ends, so the pump's next `send` fails; the pump
  closes `reply`, the engine's next send for that sequence fails, and the engine frees the slot (clears its KV cells and
  closes the sampler) instead of decoding for nobody. The cost of leaving is one more token. A client that stops
  reading, but keeps the connection open, is noticed only when Bond's writer blocks (the pump's send then times out
  after 10 s); the kernel's socket buffers can hold a short answer whole, so such a client can finish its answer
  without ever being noticed. This was not measured.
- A failure after the head went out (a decode error, or an engine that gives no piece for 60 s) cannot change the
  status: the pump sends one `data: {"error": ...}` event and ends the stream without `[DONE]`.
- `max_tokens` below 1 is a `400` on both paths (a stream's channel size comes from it).

For what this costs and where the gaps are, see [benchmark.md](benchmark.md) and [gaps.md](gaps.md).

## Files

```text
agentica.toml, agentica.lock   Bond pinned by tag (0.3.0); [native.llama] and [native.ggml]
server.agc                     the program (flags, wiring)
src/llama.agc                  FFI bindings and the wrapper
src/engine.agc                 the engine task: continuous batching
src/api.agc                    the Bond routes, the SSE pump
src/numbers.agc                `smaller`, shared by the other files
smoke.agc                      one sequence, to check the bindings
scripts/                       fetch scripts (URL, tag and sha256 recorded in each)
bench/client.py, run_all.py    the load client and the comparison runner
bench/results/                 the JSON of every run behind the tables
tests/test_api.py              end-to-end test of the routes, streaming and a client that leaves
```
