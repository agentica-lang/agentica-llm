# Gaps found in Agentica, Bond and the FFI

Each is a short bullet with a repro. The streaming work (Agentica 0.19.0-beta.1, Bond 0.2.0) re-checked `Int` `min`/`max`
and `case` as an expression, which still fail, and added the hex-literal and channel bullets; the other repros are
from Agentica 0.18.0 and were not rerun.

Closed since the first version: Bond had no streaming body (Bond 0.2.0 has `Bond.stream` and `Bond.sse`, on the
streaming bodies of `std/http`) and `Bond.json` encoded a struct twice (Bond 0.2.0 has `Bond.json_as` with a codec).

- **A project `[native]` file must be under `native/`.** `[native.llama]` with `linux = { file = "deps/llama.cpp/libllama.so.0" }`
  fails: `PKG-MANIFEST-INVALID ... not a native file ... a native file is a path in the native/ folder of the package`.
  A prebuilt library that lives elsewhere (a fetched release) needs symlinks into `native/linux/` (what
  `scripts/fetch-llama.sh` does) or an absolute `system` path that cannot be committed. A project-relative path
  anywhere in the tree would fix it.
- **No log callback.** `llama_log_set(cb, NULL)` needs a C function pointer (#851); `agentica smoke.agc` prints about
  200 KB of llama.cpp log lines to stderr at start, with no way to filter them. The same for `progress_callback`
  (model load progress) and `abort_callback`.
- **`Int` has no `min`, `max` or `clamp`.** `n_batch.min(512)` is `CHECK-UNKNOWN-METHOD: Int has no method "min"`;
  `src/numbers.agc` holds a `smaller(a, b)` for the three files that need it.
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
- **No hex literals.** `puts(0x80)` is `PARSE-UNEXPECTED-TOKEN: expected ")", found "x80"`, so the UTF-8 lead-byte
  checks in `utf8_complete` (`src/engine.agc`) are written `128`, `192`, `224` and `240`.
- **A channel cannot say "the reader left" without a send.** `closed?` is `true` only once the channel is closed *and
  drained*, and a failed `send` is `error("channel: closed")` or `error("channel: timed out")` (full), which the
  reference says not to match on. The engine sends with a timeout of 0 and treats any failure as "stop this sequence",
  which is right for both, but it learns that the client left only at the next token, and a request whose client leaves
  before its first token (the route is blocked in `recv`) still runs until that token. A `Channel.closed_by_reader?` or
  a send error that is an enum would let the engine tell the cases apart.
- **Top-level `server.agc` has no flag parser.** Every program that wants `--port N` writes the same loop over
  `System.args()` (`parse`/`apply` in `server.agc`); a small `std/args` would remove it.
- **Different batch shapes give different greedy text** (not an Agentica bug; llama.cpp float order). It is listed so
  nobody builds a test on "batched equals sequential".

See also the [overhead notes](benchmark.md#where-agentica-adds-overhead-and-what-would-remove-it) for what these gaps cost in the engine.
