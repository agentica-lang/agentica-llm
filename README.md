# agentica-llm

A prototype LLM inference server written in [Agentica](https://github.com/agentica-lang/agentica): the
[Bond](https://github.com/agentica-lang/bond) web framework in front of llama.cpp, called through the Agentica FFI
(no C code). It serves an OpenAI-shaped `POST /v1/chat/completions` (non-streaming) and `GET /health`.

## Get started

```sh
scripts/fetch-llama.sh && scripts/fetch-model.sh && agentica install
agentica server.agc --port 8080 --threads 4 --ctx 8192 --parallel 8

curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":32,"temperature":0}'
```

## Benchmark

On a 4-core laptop CPU with Qwen2.5-0.5B, it is within noise of `llama-server` of the same llama.cpp release (parity, not
a win): the model dominates. Details in [docs/benchmark.md](docs/benchmark.md).

## Docs

- [docs/running.md](docs/running.md): dependencies, server options, tests.
- [docs/architecture.md](docs/architecture.md): the engine, continuous batching, sampling, the file map.
- [docs/benchmark.md](docs/benchmark.md): setup, method, results, how to rerun, where the overhead is.
- [docs/gaps.md](docs/gaps.md): gaps found in Agentica, Bond and the FFI.

## License

Apache License 2.0. See [LICENSE](LICENSE).
