"""End-to-end check of the server: starts it on a free port and talks HTTP.

Needs the fetched deps (scripts/fetch-llama.sh, scripts/fetch-model.sh) and the agentica binary in AGENTICA
(default: `agentica` on PATH). Run from the repo root: `python3 -m unittest tests.test_api`.
"""
import http.client
import json
import os
import socket
import subprocess
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTICA = os.environ.get("AGENTICA", "agentica")
MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Name three colors."},
]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls.log = open(os.path.join(ROOT, "tests", "server.log"), "w")
        cls.proc = subprocess.Popen(
            [AGENTICA, "server.agc", "--port", str(cls.port), "--threads", "2", "--parallel", "4", "--ctx", "4096"],
            cwd=ROOT, stdout=cls.log, stderr=subprocess.STDOUT)
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                if cls.request("GET", "/health")[0] == 200:
                    return
            except OSError:
                time.sleep(0.3)
        cls.tearDownClass()
        raise RuntimeError("server did not start")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=20)
        cls.log.close()

    @classmethod
    def request(cls, method, path, payload=None):
        conn = http.client.HTTPConnection("127.0.0.1", cls.port, timeout=120)
        body = json.dumps(payload).encode() if payload is not None else None
        conn.request(method, path, body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, data

    def chat(self, **fields):
        status, data = self.request("POST", "/v1/chat/completions", dict(messages=MESSAGES, **fields))
        return status, json.loads(data) if status == 200 else data

    def test_health(self):
        status, data = self.request("GET", "/health")
        self.assertEqual((status, json.loads(data)), (200, {"status": "ok"}))

    def test_chat_is_openai_shaped_and_greedy_is_repeatable(self):
        status, first = self.chat(max_tokens=16, temperature=0)
        self.assertEqual(status, 200)
        self.assertEqual(first["object"], "chat.completion")
        self.assertEqual(first["choices"][0]["message"]["role"], "assistant")
        self.assertLessEqual(first["usage"]["completion_tokens"], 16)
        self.assertIn(first["choices"][0]["finish_reason"], ["stop", "length"])
        self.assertEqual(first["usage"]["total_tokens"], first["usage"]["prompt_tokens"] + first["usage"]["completion_tokens"])
        _, second = self.chat(max_tokens=16, temperature=0)
        self.assertEqual(first["choices"][0]["message"]["content"], second["choices"][0]["message"]["content"])

    def test_seeded_sampling_is_repeatable(self):
        _, a = self.chat(max_tokens=16, temperature=0.8, seed=7)
        _, b = self.chat(max_tokens=16, temperature=0.8, seed=7)
        self.assertEqual(a["choices"][0]["message"]["content"], b["choices"][0]["message"]["content"])

    def test_bad_requests_are_400(self):
        self.assertEqual(self.request("POST", "/v1/chat/completions", {"messages": []})[0], 400)
        self.assertEqual(self.request("POST", "/v1/chat/completions", {"nope": 1})[0], 400)
        long_prompt = [{"role": "user", "content": "word " * 3000}]
        self.assertEqual(self.request("POST", "/v1/chat/completions", {"messages": long_prompt})[0], 400)


if __name__ == "__main__":
    unittest.main()
