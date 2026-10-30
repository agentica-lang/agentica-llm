#!/usr/bin/env python3
"""Closed-loop load client for an OpenAI-style /v1/chat/completions server.

Standard library only: one thread per concurrent client, each with its own
keep-alive connection, sending requests back to back until `--requests` are
done. Prints one JSON object with requests/s, generated tokens/s and latency
percentiles.
"""
import argparse
import http.client
import json
import threading
import time
from urllib.parse import urlparse

SYSTEM = "You are a helpful assistant."
QUESTIONS = [
    "Explain how a hash table works.",
    "Write a short story about a lighthouse keeper.",
    "What are the main causes of the French Revolution?",
    "Describe the water cycle step by step.",
    "Give me a recipe for vegetable soup.",
    "How does a binary search work? Show an example.",
    "Why is the sky blue?",
    "List some tips for learning a new language.",
]


def body_for(index, max_tokens, extra):
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": QUESTIONS[index % len(QUESTIONS)]},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 1,
    }
    payload.update(extra)
    return json.dumps(payload).encode()


def percentile(sorted_values, q):
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * q
    low = int(k)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (k - low)


def run(url, concurrency, total, max_tokens, extra, keep_texts=False):
    target = urlparse(url)
    lock = threading.Lock()
    state = {"next": 0}
    latencies = []
    tokens = []
    prompt_tokens = []
    texts = {}
    failures = []

    def worker():
        conn = http.client.HTTPConnection(target.hostname, target.port, timeout=600)
        while True:
            with lock:
                index = state["next"]
                if index >= total:
                    break
                state["next"] += 1
            body = body_for(index, max_tokens, extra)
            started = time.perf_counter()
            try:
                conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
                response = conn.getresponse()
                raw = response.read()
                elapsed = time.perf_counter() - started
                if response.status != 200:
                    raise RuntimeError("status %d: %s" % (response.status, raw[:200]))
                data = json.loads(raw)
                with lock:
                    latencies.append(elapsed)
                    tokens.append(data["usage"]["completion_tokens"])
                    prompt_tokens.append(data["usage"]["prompt_tokens"])
                    if keep_texts:
                        texts[index] = data["choices"][0]["message"]["content"]
            except Exception as exc:  # a failed request is counted, the run goes on
                with lock:
                    failures.append(str(exc))
                conn.close()
                conn = http.client.HTTPConnection(target.hostname, target.port, timeout=600)
        conn.close()

    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    begin = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - begin
    ordered = sorted(latencies)
    result = {
        "concurrency": concurrency,
        "requests": len(latencies),
        "failures": len(failures),
        "wall_s": wall,
        "req_per_s": len(latencies) / wall,
        "gen_tok_per_s": sum(tokens) / wall,
        "completion_tokens": sum(tokens),
        "prompt_tokens": sum(prompt_tokens),
        "p50_s": percentile(ordered, 0.50),
        "p95_s": percentile(ordered, 0.95),
    }
    if failures:
        result["first_failure"] = failures[0]
    if keep_texts:
        result["texts"] = {str(k): texts[k] for k in sorted(texts)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--extra", default="{}", help="JSON merged into every request body")
    parser.add_argument("--texts", action="store_true", help="include the answers in the output")
    args = parser.parse_args()
    result = run(args.url, args.concurrency, args.requests, args.max_tokens, json.loads(args.extra), args.texts)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
