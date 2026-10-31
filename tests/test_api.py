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
        usage = first["usage"]
        self.assertEqual(usage["total_tokens"], usage["prompt_tokens"] + usage["completion_tokens"])
        _, second = self.chat(max_tokens=16, temperature=0)
        self.assertEqual(first["choices"][0]["message"]["content"], second["choices"][0]["message"]["content"])

    def test_seeded_sampling_is_repeatable(self):
        _, a = self.chat(max_tokens=16, temperature=0.8, seed=7)
        _, b = self.chat(max_tokens=16, temperature=0.8, seed=7)
        self.assertEqual(a["choices"][0]["message"]["content"], b["choices"][0]["message"]["content"])

    @classmethod
    def stream(cls, messages=MESSAGES, **fields):
        """POSTs with `stream: true`; returns (status, content type, body). The body ends when the server closes."""
        conn = http.client.HTTPConnection("127.0.0.1", cls.port, timeout=120)
        payload = dict(messages=messages, stream=True, **fields)
        conn.request("POST", "/v1/chat/completions", json.dumps(payload), {"Content-Type": "application/json"})
        response = conn.getresponse()
        body = response.read().decode("utf-8")
        conn.close()
        return response.status, response.getheader("Content-Type"), body

    def events(self, body):
        """The `data:` payloads of a stream body, in order."""
        blocks = [block for block in body.split("\n\n") if block]
        self.assertTrue(all(block.startswith("data: ") for block in blocks), body)
        return [block[len("data: "):] for block in blocks]

    def test_stream_chunks_add_up_to_the_greedy_answer(self):
        _, whole = self.chat(max_tokens=24, temperature=0)
        status, content_type, body = self.stream(
            max_tokens=24, temperature=0, stream_options={"include_usage": True})
        events = self.events(body)
        self.assertEqual(status, 200)
        self.assertTrue(content_type.startswith("text/event-stream"), content_type)
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(e) for e in events[:-1]]
        self.assertEqual({c["object"] for c in chunks}, {"chat.completion.chunk"})
        self.assertEqual(len({c["id"] for c in chunks}), 1)
        # first chunk: the role; then content pieces; then the finish reason; then the usage chunk (no choices)
        first, last, usage = chunks[0]["choices"][0], chunks[-2]["choices"][0], chunks[-1]
        self.assertEqual(first["delta"]["role"], "assistant")
        self.assertIsNone(first["finish_reason"])
        self.assertEqual(last["delta"], {})
        self.assertEqual(last["finish_reason"], whole["choices"][0]["finish_reason"])
        self.assertEqual(usage["choices"], [])
        self.assertEqual(usage["usage"], whole["usage"])
        middle = [c["choices"][0] for c in chunks[1:-2]]
        self.assertTrue(all(m["finish_reason"] is None and m["delta"]["content"] for m in middle))
        self.assertEqual("".join(m["delta"]["content"] for m in middle), whole["choices"][0]["message"]["content"])

    def test_stream_keeps_multibyte_characters_whole(self):
        # Emoji often span several tokens: every piece must still be valid UTF-8 (the body decodes and the pieces
        # have no replacement character) and the pieces must add up to the non-streaming text.
        messages = [{"role": "user", "content": "Write 3 lines using rare emoji such as \U0001FAE0 \U0001FABF."}]
        _, data = self.request("POST", "/v1/chat/completions",
                               {"messages": messages, "max_tokens": 60, "temperature": 0})
        _, _, body = self.stream(messages, max_tokens=60, temperature=0)
        pieces = [json.loads(e)["choices"][0]["delta"].get("content", "") for e in self.events(body)[:-1]]
        self.assertNotIn("\ufffd", "".join(pieces))
        self.assertEqual("".join(pieces), json.loads(data)["choices"][0]["message"]["content"])

    def test_stream_of_a_refused_request_is_400(self):
        long_prompt = [{"role": "user", "content": "word " * 3000}]
        self.assertEqual(self.stream(long_prompt)[0], 400)

    def test_clients_that_leave_free_their_slots(self):
        # More abandoned streams than slots (4), each asking for far more than it reads. If the engine kept
        # decoding for them, the request after them would wait minutes for a slot.
        for _ in range(6):
            sock = socket.create_connection(("127.0.0.1", self.port), timeout=120)
            body = json.dumps({"messages": MESSAGES, "stream": True, "max_tokens": 900, "temperature": 0})
            sock.sendall(("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                          "Content-Length: %d\r\n\r\n%s" % (len(body), body)).encode())
            got = b""
            while b"data: {" not in got.split(b"\r\n\r\n", 1)[-1]:
                chunk = sock.recv(4096)
                self.assertTrue(chunk, "the server closed before the first event")
                got += chunk
            sock.close()
        started = time.time()
        status, answer = self.chat(max_tokens=8, temperature=0)
        self.assertEqual(status, 200)
        self.assertLess(time.time() - started, 60)
        self.assertEqual(self.request("GET", "/health")[0], 200)

    def test_bad_requests_are_400(self):
        self.assertEqual(self.request("POST", "/v1/chat/completions", {"messages": []})[0], 400)
        self.assertEqual(self.request("POST", "/v1/chat/completions", {"nope": 1})[0], 400)
        long_prompt = [{"role": "user", "content": "word " * 3000}]
        self.assertEqual(self.request("POST", "/v1/chat/completions", {"messages": long_prompt})[0], 400)
        for bad in (0, -1, -5):
            self.assertEqual(self.chat(max_tokens=bad)[0], 400, bad)
            self.assertEqual(self.stream(max_tokens=bad)[0], 400, bad)
        self.assertEqual(self.request("GET", "/health")[0], 200)


if __name__ == "__main__":
    unittest.main()
