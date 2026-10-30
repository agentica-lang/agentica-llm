#!/usr/bin/env python3
"""Runs the Agentica server and llama-server one after the other on the same
model and settings, and prints the medians as a Markdown table.

Each server starts under a systemd scope with a memory cap, is warmed up, then
each concurrency runs `--reps` times. Processes are stopped by the PID that
this script started.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "deps", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
LLAMA_DIR = os.path.join(ROOT, "deps", "llama.cpp")
SCOPE = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=8G", "-p", "MemorySwapMax=0", "--"]


def commands(agentica, port, threads, ctx, slots):
    """The command line and environment of each server, on the same settings."""
    return {
        "agentica": (
            SCOPE
            + [agentica, "server.agc", "--port", str(port), "--threads", str(threads), "--ctx", str(ctx),
               "--parallel", str(slots), "--batch", "2048"],
            {},
        ),
        "llama-server": (
            SCOPE
            + [os.path.join(LLAMA_DIR, "llama-server"), "-m", MODEL, "--host", "127.0.0.1", "--port", str(port),
               "-t", str(threads), "-tb", str(threads), "-c", str(ctx), "-np", str(slots), "-b", "2048",
               "-ub", "512", "--no-webui", "--no-cache-prompt", "-cram", "0", "--no-warmup"],
            {},
        ),
    }


def rss_kb(pid, field):
    with open("/proc/%d/status" % pid) as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return 0


def children(pid):
    """The PIDs under `pid` (the scope may leave a wrapper in front of the server)."""
    out = subprocess.run(["ps", "-o", "pid=", "--ppid", str(pid)], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out]


def wait_ready(port, proc, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("server exited early with %s" % proc.returncode)
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError("server not ready")


class Server:
    """A server process started by this script; `stop` ends exactly that PID."""

    def __init__(self, name, cmd, env, port, log_dir):
        self.name = name
        self.port = port
        self.url = "http://127.0.0.1:%d" % port
        self.extra = {"cache_prompt": False} if name == "llama-server" else {}
        self.log = open(os.path.join(log_dir, name + ".log"), "w")
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, **env))
        self.pid = self.proc.pid  # `systemd-run --scope` execs the command: this is the server itself

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.log.close()


def load1():
    return os.getloadavg()[0]


def bench(servers, concurrencies, reps, max_tokens):
    """Runs the servers alternately, rep by rep, so a noisy machine hurts both the same."""
    for s in servers:
        wait_ready(s.port, s.proc)
    results = {s.name: {"runs": {}, "rss_idle_mb": rss_kb(s.pid, "VmRSS") / 1024} for s in servers}
    for s in servers:
        client.run(s.url, 4, 16, max_tokens, s.extra)  # warm-up: weights paged in, buffers sized
        client.run(s.url, 1, 2, max_tokens, s.extra)
        results[s.name]["texts"] = client.run(s.url, 1, 8, max_tokens, s.extra, keep_texts=True)["texts"]
        batched = client.run(s.url, 8, 8, max_tokens, s.extra, keep_texts=True)["texts"]
        # The same prompts decoded side by side must give the same greedy answers as one at a time.
        results[s.name]["batched_same"] = sum(1 for k, v in results[s.name]["texts"].items() if batched.get(k) == v)
    for c in concurrencies:
        total = max(16, 4 * c)
        for s in servers:
            results[s.name]["runs"][c] = []
        for rep in range(reps):
            order = servers if rep % 2 == 0 else servers[::-1]
            for s in order:
                before = load1()
                r = client.run(s.url, c, total, max_tokens, s.extra)
                r["load1_before"] = before
                results[s.name]["runs"][c].append(r)
                print("%s c=%d rep=%d: %.1f tok/s, load %.1f" % (s.name, c, rep, r["gen_tok_per_s"], before),
                      flush=True)
    for s in servers:
        results[s.name]["rss_peak_mb"] = rss_kb(s.pid, "VmHWM") / 1024
        results[s.name]["rss_end_mb"] = rss_kb(s.pid, "VmRSS") / 1024
    return results


def median(values):
    return statistics.median(values)


def table(results):
    lines = ["| server | conc | req/s | gen tok/s | p50 s | p95 s | failed |", "|---|---|---|---|---|---|---|"]
    for name, data in results.items():
        for c, runs in data["runs"].items():
            lines.append(
                "| %s | %s | %.2f | %.1f | %.2f | %.2f | %d |"
                % (
                    name,
                    c,
                    median([r["req_per_s"] for r in runs]),
                    median([r["gen_tok_per_s"] for r in runs]),
                    median([r["p50_s"] for r in runs]),
                    median([r["p95_s"] for r in runs]),
                    sum(r["failures"] for r in runs),
                )
            )
    lines += ["", "| server | RSS idle MB | RSS peak MB | RSS end MB |", "|---|---|---|---|"]
    for name, data in results.items():
        lines.append("| %s | %.0f | %.0f | %.0f |" % (name, data["rss_idle_mb"], data["rss_peak_mb"], data["rss_end_mb"]))
    lines += [""] + ["%s: %d of 8 answers at concurrency 8 equal the ones at concurrency 1."
                     % (n, d["batched_same"]) for n, d in results.items()]
    names = list(results)
    if len(names) == 2:
        a, b = (results[n]["texts"] for n in names)
        same = sum(1 for k in a if a[k] == b.get(k))
        lines += ["", "Greedy answers identical between servers: %d of %d prompts." % (same, len(a))]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agentica", default="agentica")
    parser.add_argument("--only", choices=["agentica", "llama-server"])
    parser.add_argument("--order", default="agentica,llama-server")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--concurrency", default="1,4,16")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--out", default=os.path.join(ROOT, "bench", "results", "latest.json"))
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    concurrencies = [int(c) for c in args.concurrency.split(",")]
    log_dir = os.path.dirname(args.out)
    servers = []
    try:
        for i, name in enumerate(args.order.split(",")):
            if args.only and name != args.only:
                continue
            port = args.port + i
            cmd, env = commands(args.agentica, port, args.threads, args.ctx, args.slots)[name]
            servers.append(Server(name, cmd, env, port, log_dir))
        results = bench(servers, concurrencies, args.reps, args.max_tokens)
    finally:
        for s in servers:
            s.stop()
    with open(args.out, "w") as f:
        json.dump({"threads": args.threads, "ctx": args.ctx, "slots": args.slots, "max_tokens": args.max_tokens,
                   "results": results}, f, indent=1)
    print(table(results))


if __name__ == "__main__":
    main()
