#!/usr/bin/env bash
# The release archive ships no headers. The struct layouts in src/llama.agc are
# copied from llama.h of the same tag; this fetches it for reference/checking.
set -euo pipefail
TAG="b12012"
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$root/deps/headers"
for f in include/llama.h ggml/include/ggml.h ggml/include/ggml-backend.h; do
  curl -fsL "https://raw.githubusercontent.com/ggml-org/llama.cpp/$TAG/$f" -o "$root/deps/headers/$(basename "$f")"
done
