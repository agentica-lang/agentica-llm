# agentica-llm

A prototype LLM inference server written in [Agentica](https://github.com/agentica-lang/agentica): the
[Bond](https://github.com/agentica-lang/bond) web framework in front of llama.cpp, called through the Agentica FFI
(no C code). It is benchmarked against `llama-server` of the same llama.cpp release.

Written for Agentica 0.18.0 and Bond at commit `a673e4a` (pinned in `agentica.toml`).

## What is in it

- `src/llama.agc`: bindings for the part of `llama.h` a server needs (backend init, model load, context, vocab,
  tokenize, chat template, batch, decode, sampler chain, token to piece, free). `llama_decode` and the model load are
  `compute def`. The `llama_*_params` structs and `llama_batch` are FFI structs passed by value.
- `src/engine.agc`: one task owns the context and does continuous batching. Requests arrive on a `Channel`; each step
  puts the next token of every active sequence (or the whole prompt of a new one) into one batch, up to
  `--parallel` sequences (each its own `seq_id`, its own `n_ctx / parallel` share of the KV cache), decodes once,
  samples every sequence with llama.cpp's sampler chain in C, and answers on a per-request channel. A freed slot takes the
  next waiting request in the next step.
- `src/api.agc`, `server.agc`: Bond routes `POST /v1/chat/completions` (OpenAI shape, non-streaming: `messages`,
  `max_tokens`, `temperature`, `seed`) and `GET /health`.
- `bench/client.py`: the load client (standard library only). `bench/run_all.py`: starts both servers and runs the
  comparison.
- `tests/test_api.py`: an end-to-end test of the routes.

Streaming (SSE) comes when `std/http` can stream (agentica #803).

## Fetch the dependencies

```sh
scripts/fetch-llama.sh     # llama.cpp b12012, Linux x86-64 CPU build, sha256-checked, into deps/llama.cpp
scripts/fetch-model.sh     # Qwen2.5-0.5B-Instruct Q4_K_M, sha256-checked, into deps/
scripts/fetch-headers.sh   # llama.h of the same tag, only to compare the struct layouts in src/llama.agc
agentica install           # Bond (git dependency, pinned by commit)
```

Nothing under `deps/` is committed. The release archive ships no headers, so the struct layouts were copied from
`llama.h` at tag `b12012`; a wrong field list is undefined behavior, so re-check them if you change the tag.
`fetch-llama.sh` also links the libraries into `native/linux/` because Agentica only loads `[native]` files from a
`native/` folder (see the gaps).

## Run the server

```sh
agentica server.agc --port 8080 --threads 4 --ctx 8192 --parallel 8
# options: --model PATH --host H --port N --threads N --ctx N --parallel N --batch N --backends DIR --stats on

curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":32,"temperature":0}'
```

llama.cpp logs a lot to stderr at start (the log callback needs an FFI callback, #851); send it to a file.
`--stats on` prints, every 5 s of work, how much of the engine's time went into `llama_decode` and how much into
Agentica.

Test: `AGENTICA=/path/to/agentica python3 -m unittest tests.test_api`.

## Run the benchmark

```sh
python3 bench/run_all.py --agentica /path/to/agentica --threads 4 --max-load 1.5
```

Both servers get the same model, thread count (`-t` and `-tb`), context (`-c 8192`), slots (`-np 8`, so 1024 tokens
of KV per slot in both), batch sizes (`-b 2048 -ub 512`), greedy sampling (`temperature` 0) and `max_tokens` 64. The
`llama-server` runs with `--no-webui --no-cache-prompt -cram 0` and every request says `cache_prompt: false`, so it
cannot skip the prompt work. Each server starts under
`systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 --` and is stopped by the PID the script started.

`bench/client.py` is closed-loop: N threads, each with a keep-alive connection, send requests back to back (16
requests at concurrency 1 and 4, 64 at concurrency 16). Eight different prompts rotate; the system prompt is explicit
so both chat templates give the same prompt (27.75 prompt tokens per request on average in both servers). After a warm-up, each
concurrency runs 3 times per server, **alternating the servers rep by rep** so a noisy minute hurts both, and the
tables show medians. `--max-load N` makes the script wait until the neighbours use fewer than N cores before each run
(the busy cores are stored with each run).

## Results

Machine: AMD Ryzen 5 3550H (4 cores, 8 threads, a laptop), 12 GB, Linux 6.12. **It is a shared machine**: other
agents ran test suites on it during all runs (the runner waited for under 1.5 busy neighbour cores, and most runs saw
0.2 to 0.5). Treat differences under about 10 % as noise: the three reps of one cell often differ by more than that
(see `bench/results/*.json` for every rep).

Model: Qwen2.5-0.5B-Instruct Q4_K_M. llama.cpp b12012 (CPU). 8 slots, 8192 context, 64 max tokens, temperature 0.
Medians of 3 runs.

### 4 threads (one per physical core)

| server | concurrency | req/s | generated tok/s | p50 latency s | p95 latency s |
|---|---|---|---|---|---|
| agentica-llm | 1 | 0.50 | 31.6 | 1.83 | 2.46 |
| llama-server | 1 | 0.55 | 35.1 | 1.74 | 2.01 |
| agentica-llm | 4 | 0.65 | 41.8 | 6.06 | 6.64 |
| llama-server | 4 | 0.62 | 39.7 | 6.47 | 7.03 |
| agentica-llm | 16 | 0.67 | 42.8 | 23.69 | 25.82 |
| llama-server | 16 | 0.66 | 42.1 | 23.67 | 28.96 |

### 8 threads (`nproc`)

| server | concurrency | req/s | generated tok/s | p50 latency s | p95 latency s |
|---|---|---|---|---|---|
| agentica-llm | 1 | 0.44 | 27.7 | 2.25 | 2.57 |
| llama-server | 1 | 0.48 | 30.8 | 2.07 | 2.26 |
| agentica-llm | 4 | 0.54 | 34.7 | 7.40 | 7.99 |
| llama-server | 4 | 0.49 | 31.1 | 8.25 | 9.28 |
| agentica-llm | 16 | 0.54 | 34.6 | 28.96 | 32.69 |
| llama-server | 16 | 0.35 | 22.4 | 42.67 | 74.19 |

No request failed in any run. (The latency at concurrency 16 is mostly queueing: 16 clients for 8 slots.)

### Memory (RSS)

| server | idle after load | peak (VmHWM) | end of run |
|---|---|---|---|
| agentica-llm | 689 MB | 782 MB | 757 MB |
| llama-server | 671 MB | 703 MB | 703 MB |

### The serving layer alone (`GET /health`, 3000 requests, both servers pinned to CPUs 0 to 3)

| server | concurrency | req/s | p50 ms | p95 ms |
|---|---|---|---|---|
| agentica-llm (Bond on `std/http`) | 1 | 2426 | 0.35 | 0.53 |
| llama-server | 1 | 4320 | 0.21 | 0.26 |
| agentica-llm | 16 | 5747 | 1.79 | 7.94 |
| llama-server | 16 | 6289 | 1.27 | 8.26 |

(One Python client process produces most of that ceiling.)

### Correctness checks

- Greedy answers of the two servers are identical for all 8 prompts at concurrency 1 (64 tokens each).
- Decoded side by side (concurrency 8), answers drift from the one-at-a-time ones late in the text (after 60 to 290
  characters) in both servers (agentica-llm 0 to 1 of 8 identical, llama-server 0 to 2 of 8): the batch shape changes
  the float reduction order of the quantized kernels. It is not cross-sequence contamination, which would diverge at
  once.

## An honest reading

- **Parity, not a win.** With 4 threads agentica-llm is within noise of `llama-server` at concurrency 4 and 16 and
  about 10 % behind at concurrency 1 (31.6 vs 35.1 tok/s; p50 1.83 vs 1.74 s). With 8 threads on 4 physical cores both
  are slower, and `llama-server` fell further at concurrency 16 (22 tok/s, one rep at 22, one at 21, one at 37),
  which looks like it suffers more from the neighbours' load, not like a real gap. Do not read the 8-thread
  concurrency-16 row as "Agentica is 50 % faster".
- **The model dominates.** 0.5B Q4 on 4 cores generates about 30 to 45 tok/s whatever sits in front of it, and
  concurrency barely raises it: CPU decode here is compute-bound, so batching gains little. The same would not hold on
  a GPU, where the serving layer's share of the step matters far more.
- **Agentica's share of the engine step is small.** `--stats on` (concurrency 16, CPU-limited run) shows 30 to 85 ms of
  Agentica work per 5 s window against 5.0 to 6.1 s inside `llama_decode`: about 1 to 1.5 %, around 3 to 4 ms per
  step (fill the batch, sample, token to piece, bookkeeping for 8 sequences). At 30 ms or more per step on CPU that is
  invisible, and a 1 to 3 ms floor per step is what a GPU run (steps of a few ms) would feel.
- **The HTTP layer is not the bottleneck either**: Bond on `std/http` answers 2.4k to 5.7k `/health` requests per second
  against 4.3k to 6.3k for `llama-server`, when inference gives 0.5 requests per second. At concurrency 1 it is about
  half as fast per request (0.35 vs 0.21 ms p50), which is real but about 0.1 ms on a request that takes two seconds.
- **Memory:** about 20 MB more at idle and 80 MB more at peak than `llama-server` (the Agentica VM, the Bond app, and
  the batch/scratch buffers).
- Not measured: time to first token (no streaming yet), long prompts (prefill), sampling with temperature (the C
  sampler is the same code in both, but the sampler chain here is only `temp` then `dist`, not `llama-server`'s default
  `top-k`/`top-p`/`min-p`), and any GPU build.

## Where Agentica adds overhead, and what would remove it

- **Per-token FFI writes into the batch.** `fill` writes the token, position and `n_seq_id` arrays with bulk
  `put_array_of_int32`, but the per-token `seq_id` cell still costs a `get_pointer` and a `put_int32` per token, and a
  prefill of N tokens builds `List`s of N items (a boxed `Int` each, plus `BatchItem` structs) before the copy. Compact
  `Int`/`Float` lists (#396) and a bulk way to write one int32 into many pointers would cut it.
- **Logits stay in C on purpose.** Sampling is done by llama.cpp's sampler chain (`llama_sampler_sample`); walking
  151k floats per token in Agentica would be orders of magnitude slower, and `Native.get_array_of_float32` would
  build a 151k-item `List[Float]` per row (this is where #396 matters most for any sampler written in Agentica).
- **No streaming.** The engine produces a token per step but `std/http` sends a whole response (#803), so
  time-to-first-token equals total latency; there is no way to give Bond a streaming body yet.
- **Sampling runs on a scheduler thread.** `llama_sampler_sample` is a plain `def`; with 16 sequences it blocks that
  scheduler for the sum of the calls. It is short (the stats show about 3 ms per step), but a `compute def` for the
  whole "sample every row" loop, or a C callback that samples inside `llama_decode`, would be free of it
  (needs FFI callbacks, #851).
- **llama.cpp's log goes to stderr unchecked**, because `llama_log_set` needs a callback (#851).
- **One engine task, one decode at a time.** That is how `llama.cpp` works on one context, so it is no loss, but
  there is no second context to overlap prefill and decode.

## Gaps found in Agentica, Bond and the FFI

Each is a short bullet with a repro; the numbers are for Agentica 0.18.0.

- **A project `[native]` file must be under `native/`.** `[native.llama]` with `linux = { file = "deps/llama.cpp/libllama.so.0" }`
  fails: `PKG-MANIFEST-INVALID ... not a native file ... a native file is a path in the native/ folder of the package`.
  A prebuilt library that lives elsewhere (a fetched release) needs symlinks into `native/linux/` (what
  `scripts/fetch-llama.sh` does) or an absolute `system` path that cannot be committed. A project-relative path
  anywhere in the tree would fix it.
- **No log callback.** `llama_log_set(cb, NULL)` needs a C function pointer (#851); `agentica smoke.agc` prints about
  200 KB of llama.cpp log lines to stderr at start, with no way to filter them. The same for `progress_callback`
  (model load progress) and `abort_callback`.
- **`Int` has no `min`, `max` or `clamp`.** `n_batch.min(512)` is `CHECK-UNKNOWN-METHOD: Int has no method "min"`;
  I wrote a private `smaller(a, b)` in two files.
- **An entry file in a subfolder cannot import the project's `src/`.** `tools/smoke.agc` with `import Llama from
  "src/llama"` fails with `CHECK-IMPORT-IO: cannot resolve import`; imports resolve relative to the importing file's
  folder, so every runnable program has to sit at the repo root (or the files need `../`, which this form does not
  allow). The diagnostic does not say what the path was resolved against.
- **A defs-only `extern` block has to come after every block that has a resource or a struct**, in the same file
  (`CHECK-DECL-ORDER: structs, enums, type aliases and resources come before functions`), so the one-line `ggml`
  block had to move to the end of `src/llama.agc` and cannot sit by the library it belongs to.
- **`case` is not an expression.** `x = case r ... ok(n) -> n ... end` is `PARSE-UNEXPECTED-TOKEN: expected an
  expression, found "case"`, so unwrapping a `Result` with a fallback takes a `case` statement that assigns in each arm
  (and `try` only works in a function that returns a `Result`).
- **No bulk write of a small-int list into native memory.** Only `put_array_of_int32/int64/float32/float64`
  exist; the `int8` `logits` array of `llama_batch` goes through `Bytes.from_list` and `put_bytes`, and the
  `seq_id` pointer-to-int cells need a `get_pointer` and a `put_int32` per token.
- **Bond: no streaming body.** `Bond.text`, `Bond.json` and `Conn.response_body` are whole strings, so SSE needs
  `std/http` first (#803).
- **Bond: `Bond.json` takes a `Json`, not a struct**, so `Bond.json(conn, 200, JSON.from(response))` encodes the
  value twice (`JSON.from`, then `JSON.encode` inside). `Bond.header(Bond.text(conn, 200, JSON.encode(response)),
  "content-type", "application/json")` is the way to skip one step; a `Bond.json` that took the encoded string or any
  encodable value would remove it.
- **Top-level `server.agc` has no flag parser.** Every program that wants `--port N` writes the same loop over
  `System.args()` (`parse`/`apply` in `server.agc`); a small `std/args` would remove it.
- **Different batch shapes give different greedy text** (not an Agentica bug; llama.cpp float order). It is listed so
  nobody builds a test on "batched equals sequential".

## Layout

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
tests/test_api.py              end-to-end test
```
