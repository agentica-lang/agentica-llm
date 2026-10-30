# Gaps found in Agentica, Bond and the FFI

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

See also the [overhead notes](benchmark.md#where-agentica-adds-overhead-and-what-would-remove-it) for what these gaps cost in the engine.
