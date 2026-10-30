#!/usr/bin/env bash
# Fetches Qwen2.5-0.5B-Instruct Q4_K_M (GGUF) into deps/. Never committed.
set -euo pipefail
URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
SHA256="74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/deps/qwen2.5-0.5b-instruct-q4_k_m.gguf"
mkdir -p "$root/deps"
if [ -f "$out" ] && echo "$SHA256  $out" | sha256sum -c - >/dev/null 2>&1; then
  echo "already fetched: $out"; exit 0
fi
curl -fL -o "$out" "$URL"
echo "$SHA256  $out" | sha256sum -c -
