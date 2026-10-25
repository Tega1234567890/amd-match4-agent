"""Answer-extraction backends.

Order tried by get_backend():
  1. Resident daemon on a unix socket (M4_SOCKET, default /tmp/m4_llm.sock) -- started by `--index`
     so the model is loaded ONCE, not once per question process.
  2. Local transformers model from M4_MODEL_DIR (default /models/llm), on GPU if available
     (ROCm PyTorch exposes the GPU as "cuda").
  3. OpenAI-compatible endpoint (LLM_BASE_URL / LLM_MODEL / LLM_API_KEY) -- dev/testing only;
     the official image must not ship keys and must use the GPU.
  4. None -> caller falls back to the heuristic extractor.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time

SOCK = os.environ.get("M4_SOCKET", "/tmp/m4_llm.sock")
MODEL_DIR = os.environ.get("M4_MODEL_DIR", "/models/llm")
MAX_NEW = int(os.environ.get("M4_MAX_NEW_TOKENS", "48"))

SYSTEM = (
    "You extract exact answers from web page evidence. "
    "Reply with ONLY a JSON object: {\"answer\": \"<value>\", \"evidence\": <number of the evidence block>}. "
    "The answer must be the shortest exact value (a number, date, name, place, version...), copied from the evidence, "
    "with no extra words and no full sentence. Keep units only if the question asks for them. "
    "If the evidence does not contain the answer, reply {\"answer\": \"\", \"evidence\": 0}. Never guess from memory."
)


def build_messages(question, evidence):
    blocks = "\n\n".join(f"[{i}] (from {u})\n{c}" for i, (u, c) in enumerate(evidence, 1))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Evidence:\n{blocks}\n\nQuestion: {question}\nJSON:"}]


def parse_reply(text):
    text = (text or "").strip()
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            d = json.loads(text[s:e + 1])
            return str(d.get("answer", "")).strip(), int(d.get("evidence") or 0)
        except Exception:
            pass
    first = text.splitlines()[0].strip() if text else ""
    return ("" if "not found" in first.lower() else first), 0


class LocalHF:
    name = "local-hf"

    def __init__(self, path=MODEL_DIR):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(self.device).eval()
        print(f"[llm] loaded {path} on {self.device}", file=sys.stderr)

    def complete(self, messages):
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


class OpenAICompat:
    name = "openai-compatible"

    def __init__(self):
        self.base, self.model = os.environ["LLM_BASE_URL"].rstrip("/"), os.environ["LLM_MODEL"]
        self.key = os.environ.get("LLM_API_KEY", "none")

    def complete(self, messages):
        import requests
        r = requests.post(self.base + "/chat/completions", headers={"Authorization": f"Bearer {self.key}"},
                          json={"model": self.model, "messages": messages, "temperature": 0,
                                "max_tokens": MAX_NEW}, timeout=25)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class SocketClient:
    name = "daemon"

    def complete(self, messages):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(28)
            s.connect(SOCK)
            s.sendall(json.dumps(messages).encode() + b"\n")
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
        d = json.loads(buf)
        if "error" in d:
            raise RuntimeError(d["error"])
        return d["text"]


def daemon_alive():
    if not os.path.exists(SOCK):
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(2); s.connect(SOCK)
        return True
    except OSError:
        return False


def get_backend():
    if os.environ.get("M4_NO_LLM"):
        return None
    if daemon_alive():
        return SocketClient()
    if os.path.isdir(MODEL_DIR):
        try:
            return LocalHF()
        except Exception as e:
            print(f"[llm] local load failed: {e}", file=sys.stderr)
    if os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_MODEL"):
        return OpenAICompat()
    return None


def serve():
    """Resident model server: one JSON line (messages) in, one JSON line ({text}) out."""
    import socketserver
    backend = LocalHF()

    class H(socketserver.StreamRequestHandler):
        def handle(self):
            line = self.rfile.readline()
            if not line.strip():  # liveness probe
                return
            try:
                msgs = json.loads(line)
                out = {"text": backend.complete(msgs)}
            except Exception as e:
                out = {"error": repr(e)}
            self.wfile.write(json.dumps(out).encode() + b"\n")

    if os.path.exists(SOCK):
        os.unlink(SOCK)
    with socketserver.UnixStreamServer(SOCK, H) as srv:
        print(f"[llm] serving on {SOCK}", file=sys.stderr)
        srv.serve_forever()


def start_daemon(wait_s=540):
    """Called by --index: fork the resident server and wait until it answers."""
    if daemon_alive() or not os.path.isdir(MODEL_DIR):
        return daemon_alive()
    import subprocess
    log = open(os.environ.get("M4_DAEMON_LOG", "/tmp/m4_llm.log"), "a")
    subprocess.Popen([sys.executable, os.path.abspath(__file__), "--serve"], stdout=log, stderr=log,
                     start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if daemon_alive():
            return True
        time.sleep(1)
    return False


if __name__ == "__main__" and "--serve" in sys.argv:
    serve()
