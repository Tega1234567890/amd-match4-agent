#!/usr/bin/env python3
"""Run the agent CLI (one process per question, like the harness) and score with MC-style normalisation."""
import json, os, re, subprocess, sys, time, unicodedata
from pathlib import Path

HERE = Path(__file__).parent
APP = HERE.parent / "app" / "app.py"
OUT = Path(os.environ.get("M4_OUTPUT_DIR", "/tmp/m4_out"))


def norm(s):
    s = unicodedata.normalize("NFKC", s or "").upper()
    s = re.sub(r"\b(M|METRES|METERS|C|°C|DEGREES)\b", "", s)
    return re.sub(r"[\s\-\.·_,°]", "", s)


def main():
    qs = json.loads((HERE / "questions.json").read_text())["queries"]
    rows, ok = [], 0
    for q in qs:
        cmd = [sys.executable, str(APP), "--query-id", q["id"], "--query", q["query"], "--output-dir", str(OUT)]
        if q["url"]:
            cmd += ["--url", q["url"]]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        dt = time.time() - t0
        try:
            got = json.loads((OUT / f"{q['id']}_output.json").read_text())
        except Exception:
            got = {"answer": "<NO FILE>", "source": ""}
        want = [q["expected"]] + q["aliases"]
        hit = norm(got["answer"]) in {norm(w) for w in want}
        ok += hit
        mode = re.findall(r"mode=(\S+)", p.stderr)
        rows.append((q["id"], hit, got["answer"], q["expected"], got.get("source", ""), round(dt, 1), mode[-1] if mode else "?"))
        print(f"{q['id']} {'PASS' if hit else 'FAIL'} {dt:5.1f}s got={got['answer']!r} want={q['expected']!r} src={got.get('source','')} mode={rows[-1][-1]}", flush=True)
    print(f"\nSCORE {ok}/{len(qs)}")
    return rows, ok


if __name__ == "__main__":
    main()
