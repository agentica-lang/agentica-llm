# Benchmark

agentica-llm against `llama-server` of the same llama.cpp release (b12012), on the same model and settings, with and
without streaming. The raw JSON of every run is in [bench/results/](../bench/results/): `threads4.json`,
`threads8.json` (one JSON answer per request) and `threads4-stream.json`, `threads8-stream.json` (`stream: true`, with
the time to first token). Every rep is there, with the cores the neighbours used before and during it.

Versions: Agentica 0.19.0-beta.1, Bond 0.2.0, llama.cpp b12012, measured on 2026-10-30 and 2026-10-31.

## The machine, and what the numbers mean

AMD Ryzen 5 3550H (4 cores, 8 threads, a laptop), 12 GB, Linux 6.12. **The CPU is capped at 1.4 GHz**
(`scaling_max_freq` 1400000 of 2100000, governor `schedutil`, 8 cores) to keep the machine quiet. **Absolute numbers are
therefore lower than at full speed** (about 18 to 23 generated tokens per second here, against 30 to 45 on an earlier,
uncapped run). Both servers ran back to back in the same sessions under the same cap, so **the ratios between the two
servers are what matter**; do not compare these absolute numbers with the 0.18.0 results kept in
`bench/results/0.18.0/` (a different machine state and different versions).

It is a **shared machine**: other agents run builds and test suites on it. The runner (`bench/run_all.py`) waits for
fewer than 1.5 busy neighbour cores before each run, measures the cores the neighbours used *during* the run (the
machine's busy cores minus what the server and the client used), and redoes a run once when that was over 1.2.
Neighbour cores during the kept runs: median 0.35 to 0.43, and the worst runs are listed below. Treat differences
under about 10 % as noise.

## Method

Both servers get the same model (Qwen2.5-0.5B-Instruct Q4_K_M), thread count (`-t` and `-tb`; 4 and 8 below), context
(`-c 8192`), slots (`-np 8`, so 1024 tokens of KV per slot in both), batch sizes (`-b 2048 -ub 512`), greedy sampling
(`temperature` 0) and `max_tokens` 64. The `llama-server` runs with `--no-webui --no-cache-prompt -cram 0` and every
request says `cache_prompt: false`, so it cannot skip the prompt work. Each server starts under
`systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 --` (a cap far above the 0.8 GB they use, so it does
not touch the timing) and is stopped by the PID the script started.

`bench/client.py` is closed-loop: N threads send requests back to back (16 requests at concurrency 1 and 4, 64 at
concurrency 16). Eight different prompts rotate; the system prompt is explicit so both chat templates give the same
prompt (about 28 prompt tokens per request). After a warm-up, each concurrency runs 3 times per server, **alternating
the servers rep by rep**, and the tables show medians of the 3.

With `--stream` every request says `"stream": true` and `stream_options.include_usage` (both servers accept it) and the
client records the time to first token (TTFT): the time until the first event whose `delta.content` is not empty. The
role-only first chunk does not count. A stream gets a new connection per request on both servers, because
`llama-server` closes the connection after a stream although it says keep-alive. Tokens come from the usage chunk.
(`llama-server` counts the end-of-answer token in `completion_tokens`, agentica-llm does not; with `max_tokens` 64 nearly
every answer ends at the limit, so the tok/s are comparable.)

```sh
python3 bench/run_all.py --agentica /path/to/agentica --threads 4 --max-load 1.5 --max-load-during 1.2 --attempts 2 --patience 300
python3 bench/run_all.py ... --stream      # adds time to first token
```

## Results without streaming

Medians of 3 runs. Ratio = agentica-llm / llama-server.

### 4 threads (one per physical core)

| server | concurrency | req/s | generated tok/s | p50 latency s | p95 latency s |
|---|---|---|---|---|---|
| agentica-llm | 1 | 0.29 | 18.7 | 3.38 | 3.58 |
| llama-server | 1 | 0.30 | 19.3 | 3.30 | 3.45 |
| agentica-llm | 4 | 0.36 | 23.3 | 10.97 | 11.49 |
| llama-server | 4 | 0.37 | 23.7 | 10.75 | 11.15 |
| agentica-llm | 16 | 0.36 | 23.1 | 44.16 | 45.54 |
| llama-server | 16 | 0.36 | 23.3 | 43.96 | 44.83 |

Ratios of tok/s: 0.97, 0.98, 0.99.

### 8 threads (`nproc`)

| server | concurrency | req/s | generated tok/s | p50 latency s | p95 latency s |
|---|---|---|---|---|---|
| agentica-llm | 1 | 0.25 | 15.6 | 4.03 | 4.55 |
| llama-server | 1 | 0.24 | 15.5 | 4.09 | 4.59 |
| agentica-llm | 4 | 0.32 | 20.8 | 12.53 | 12.58 |
| llama-server | 4 | 0.28 | 18.1 | 14.19 | 14.40 |
| agentica-llm | 16 | 0.33 | 21.4 | 47.61 | 48.67 |
| llama-server | 16 | 0.34 | 21.5 | 47.51 | 48.43 |

Ratios of tok/s: 1.01, 1.15, 1.00. The concurrency 4 row is reproducible (its three reps are 20.3, 20.9, 20.8 against
18.0, 18.1, 18.3), but the same gap is not there with 4 threads, and it is the only cell with a gap; I do not know why,
and I would not call it a win without more runs and a look at `llama-server`'s scheduling.

No request failed in any run. (The latency at concurrency 16 is mostly queueing: 16 clients for 8 slots.)

## Results with streaming (`stream: true` on both servers)

Medians of 3 runs. TTFT is the time to the first content token; total latency is until `data: [DONE]`.

### 4 threads

| server | concurrency | generated tok/s | TTFT p50 s | TTFT p95 s | total p50 s | total p95 s |
|---|---|---|---|---|---|---|
| agentica-llm | 1 | 18.3 | 0.62 | 0.70 | 3.49 | 3.60 |
| llama-server | 1 | 18.7 | 0.61 | 0.73 | 3.39 | 3.59 |
| agentica-llm | 4 | 22.7 | 2.49 | 2.66 | 11.29 | 11.50 |
| llama-server | 4 | 20.6 | 2.38 | 2.73 | 12.42 | 12.71 |
| agentica-llm | 16 | 18.8 | 31.88 | 38.84 | 52.81 | 63.12 |
| llama-server | 16 | 15.6 | 37.54 | 40.32 | 65.06 | 68.83 |

The concurrency 16 rows are **noisy**: a neighbour used 1 to 3 cores during some reps (the tok/s of the three reps are
18.8, 13.4, 19.0 for agentica-llm and 15.6, 12.5, 21.4 for `llama-server`), so read them as "the same", not as a win.
The same for one rep of concurrency 4 (19.7 against 22.7, 22.7).

### 8 threads

| server | concurrency | generated tok/s | TTFT p50 s | TTFT p95 s | total p50 s | total p95 s |
|---|---|---|---|---|---|---|
| agentica-llm | 1 | 15.5 | 0.67 | 0.77 | 4.08 | 4.53 |
| llama-server | 1 | 15.7 | 0.67 | 0.78 | 4.02 | 4.55 |
| agentica-llm | 4 | 20.6 | 2.53 | 2.96 | 12.39 | 12.88 |
| llama-server | 4 | 17.2 | 2.01 | 2.87 | 14.81 | 15.86 |
| agentica-llm | 16 | 21.4 | 28.01 | 28.65 | 47.82 | 48.23 |
| llama-server | 16 | 21.6 | 27.81 | 28.62 | 47.21 | 48.48 |

(One agentica-llm rep of concurrency 4 ran at 6.7 tok/s with a neighbour at 1.3 cores; the median ignores it.)

No request failed in any run.

### Memory (RSS, from the 4-thread runs)

| server | idle after load | peak (VmHWM), plain | peak, streaming | end of run, plain |
|---|---|---|---|---|
| agentica-llm | 691 MB | 799 MB | 801 MB | 765 MB |
| llama-server | 671 MB | 705 MB | 707 MB | 705 MB |

### The serving layer alone (`GET /health`, 3000 requests, both servers pinned to CPUs 0 to 3 with `taskset`)

Medians of 3 runs.

| server | concurrency | req/s | p50 ms | p95 ms |
|---|---|---|---|---|
| agentica-llm (Bond 0.2.0 on `std/http`) | 1 | 1734 | 0.52 | 0.64 |
| llama-server | 1 | 2454 | 0.34 | 0.45 |
| agentica-llm | 16 | 2888 | 3.35 | 16.02 |
| llama-server | 16 | 2733 | 2.97 | 18.43 |

(One Python client process produces most of that ceiling, and the CPUs are capped.)

### Correctness checks

- Greedy answers of the two servers are identical for all 8 prompts at concurrency 1 (64 tokens each), in all four
  runs, streamed or not.
- The streamed text equals the non-streaming greedy text (`tests/test_api.py`, with multi-byte emoji that split
  across tokens).
- Decoded side by side (concurrency 8), answers drift from the one-at-a-time ones late in the text in both servers
  (agentica-llm 1 to 2 of 8 identical, llama-server 0 to 1 of 8): the batch shape changes the float reduction order of
  the quantized kernels. It is not cross-sequence contamination, which would diverge at once.

## An honest reading

- **Parity, not a win.** Without streaming, agentica-llm is within 3 % of `llama-server` in 5 of 6 cells and ahead
  by 15 % in one (8 threads, concurrency 4). With streaming the same: equal at concurrency 1, equal to ahead at 4 and
  16, and the cells where it is ahead are the ones that look like noise or like `llama-server` suffering (see the notes
  under each table).
- **Streaming is free here.** At concurrency 1 the tok/s and the total latency with `stream: true` are within about 3 %
  of the plain runs on both servers. The time to first token at concurrency 1 is 0.6 to 0.7 s on both: one prefill of
  about 28 tokens plus the first decode step at 1.4 GHz. It is the same on the two servers, so Agentica's layers (the
  route, the channel, the pump, Bond's SSE, `std/http`) add nothing visible to it.
- **TTFT grows with the number of requests in flight, not with the server.** At concurrency 4 it is about 2.0 to
  2.5 s (four prompts prefilled together and decoded among the others' steps); at concurrency 16 it is 28 to 38 s
  because 8 requests wait for a slot. That is the closed-loop test with more clients than slots, not a property of the
  streaming path.
- **The model dominates.** 0.5B Q4 on 4 cores generates about 15 to 23 tok/s here whatever sits in front of it, and
  concurrency barely raises it: CPU decode is compute-bound, so batching gains little. The same would not hold on a
  GPU, where the serving layer's share of the step matters far more.
- **Agentica's share of the engine step is small** (measured with `--stats on` on 0.18.0 at full speed, not rerun
  here): 30 to 85 ms of Agentica work per 5 s window against 5.0 to 6.1 s inside `llama_decode`, about 1 to 1.5 %,
  around 3 to 4 ms per step. Under the 1.4 GHz cap the decode is slower and the share smaller. The streaming path adds
  a channel send per token to the step.
- **The HTTP layer is not the bottleneck either**: `/health` is about 30 % slower per request at concurrency 1 (0.52
  against 0.34 ms p50) and on par at concurrency 16, when inference gives 0.3 requests per second.
- **Memory:** about 20 MB more at idle and about 95 MB more at peak than `llama-server` (the Agentica VM, the Bond app,
  and the batch/scratch buffers); streaming adds about 2 MB.
- Not measured: long prompts (prefill), sampling with temperature (the C sampler is the same code in both, but the
  sampler chain here is only `temp` then `dist`, not `llama-server`'s default `top-k`/`top-p`/`min-p`), a client that
  leaves mid-stream under load, and any GPU build.

## Older runs

`bench/results/0.18.0/` holds the JSON of the first version (Agentica 0.18.0, Bond at a commit, no streaming) on the
uncapped CPU; its conclusion was the same (parity). `bench/results/first-pass/` is a 4-thread run of this version
that was thrown away: the runner then checked the neighbours only before a run, and a neighbour that started during one
spoiled it (the same cell ranged from 15 to 34 tok/s). That is why the runner now measures the neighbours during the run.

## Where Agentica adds overhead, and what would remove it

- **Per-token FFI writes into the batch.** `fill` writes the token, position and `n_seq_id` arrays with bulk
  `put_array_of_int32`, but the per-token `seq_id` cell still costs a `get_pointer` and a `put_int32` per token, and a
  prefill of N tokens builds `List`s of N items (a boxed `Int` each, plus `BatchItem` structs) before the copy. Compact
  `Int`/`Float` lists (#396) and a bulk way to write one int32 into many pointers would cut it.
- **Logits stay in C on purpose.** Sampling is done by llama.cpp's sampler chain (`llama_sampler_sample`); walking
  151k floats per token in Agentica would be orders of magnitude slower, and `Native.get_array_of_float32` would
  build a 151k-item `List[Float]` per row (this is where #396 matters most for any sampler written in Agentica).
- **Streaming costs a little per token, and it does not show.** Each token is a channel send from the engine, a
  `JSON.encode` of the chunk and a `Bond.sse_event` in the pump task, and a channel send to Bond. The engine does only the
  first part inside its step (a `send` with timeout 0, not timed separately); the rest runs in other tasks. Tokens per
  second with `stream: true` match the non-streaming runs (see the tables).
- **Sampling runs on a scheduler thread.** `llama_sampler_sample` is a plain `def`; with 16 sequences it blocks that
  scheduler for the sum of the calls. It is short (the stats show about 3 ms per step), but a `compute def` for the
  whole "sample every row" loop, or a C callback that samples inside `llama_decode`, would be free of it
  (needs FFI callbacks, #851).
- **llama.cpp's log goes to stderr unchecked**, because `llama_log_set` needs a callback (#851).
- **One engine task, one decode at a time.** That is how `llama.cpp` works on one context, so it is no loss, but
  there is no second context to overlap prefill and decode.
