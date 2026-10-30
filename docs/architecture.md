# Architecture

The [Bond](https://github.com/agentica-lang/bond) web framework in front of llama.cpp, called through the Agentica FFI
(no C code). Written for Agentica 0.18.0 and Bond at commit `a673e4a` (pinned in `agentica.toml`).

## How it works

- [src/llama.agc](../src/llama.agc): bindings for the part of `llama.h` a server needs (backend init, model load, context, vocab,
  tokenize, chat template, batch, decode, sampler chain, token to piece, free). `llama_decode` and the model load are
  `compute def`. The `llama_*_params` structs and `llama_batch` are FFI structs passed by value.
- [src/engine.agc](../src/engine.agc): one task owns the context and does continuous batching. Requests arrive on a `Channel`; each step
  puts the next token of every active sequence (or the whole prompt of a new one) into one batch, up to
  `--parallel` sequences (each its own `seq_id`, its own `n_ctx / parallel` share of the KV cache), decodes once,
  samples every sequence with llama.cpp's sampler chain in C, and answers on a per-request channel. A freed slot takes the
  next waiting request in the next step.
- [src/api.agc](../src/api.agc), [server.agc](../server.agc): Bond routes `POST /v1/chat/completions` (OpenAI shape, non-streaming: `messages`,
  `max_tokens`, `temperature`, `seed`) and `GET /health`.

Streaming (SSE) comes when `std/http` can stream (agentica #803).

For what this costs and where the gaps are, see [benchmark.md](benchmark.md) and [gaps.md](gaps.md).

## Files

```text
agentica.toml, agentica.lock   Bond pinned by commit; [native.llama] and [native.ggml]
server.agc                     the program (flags, wiring)
src/llama.agc                  FFI bindings and the wrapper
src/engine.agc                 the engine task: continuous batching
src/api.agc                    the Bond routes
smoke.agc                      one sequence, to check the bindings
scripts/                       fetch scripts (URL, tag and sha256 recorded in each)
bench/client.py, run_all.py    the load client and the comparison runner
bench/results/                 the JSON of every run behind the tables
tests/test_api.py              end-to-end test of the routes
```
