"""I/O contract layer -- the ONLY file that should need editing when AMD publishes the Match 4 spec.

ASSUMED contract (unofficial; modelled on Mini-Challenge 3 + public Match 4 project pages, 25 Oct 2026):

  Warm-up, once, before any graded clock (loads the model, starts the resident daemon):
      python3 /app/app.py --index [/app/corpus]

  One process per question:
      python3 /app/app.py --query-id query_01 --query "What year was X founded?" [--url https://start.page/]

  Optional batch mode (one JSON file of questions in, one output file per item):
      python3 /app/app.py --input /app/input/questions.json
        questions.json = [{"id": "query_01", "query": "...", "url": "https://..."}, ...]
        (also accepts {"queries": [...]}, and keys question/start_url/query_id)

  Output, always written (even on crash -> empty answer):
      /app/output/<query-id>_output.json
      {"answer": "1889", "source": "https://...", "citations": ["https://..."], "confidence": 0.82}

  answer      value only, no prose; "" when not found (refusal)
  source      URL of the page the answer was read from; "" when refusing
  citations   [source] or [] -- kept so an MC3-style grader that wants a list also works
  confidence  recorded only
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("M4_OUTPUT_DIR", "/app/output"))
OUTPUT_TEMPLATE = os.environ.get("M4_OUTPUT_TEMPLATE", "{qid}_output.json")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="AMD AI League Match 4 web retrieval agent")
    ap.add_argument("--index", nargs="?", const="", default=None,
                    help="warm-up pass (load model / start daemon); optional corpus arg ignored")
    ap.add_argument("--query", "--question", dest="query")
    ap.add_argument("--query-id", "--id", dest="query_id", default="query_01")
    ap.add_argument("--url", "--start-url", dest="url", default=None)
    ap.add_argument("--corpus", default=None, help="accepted for MC3 compatibility; unused")
    ap.add_argument("--input", default=None, help="batch JSON file of questions")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--stdout", action="store_true", help="also print the JSON to stdout")
    return ap.parse_args(argv)


def load_batch(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("queries") or data.get("questions") or data.get("items") or []
    items = []
    for i, d in enumerate(data, 1):
        if isinstance(d, str):
            d = {"query": d}
        items.append({
            "id": str(d.get("id") or d.get("query_id") or f"query_{i:02d}"),
            "query": d.get("query") or d.get("question") or "",
            "url": d.get("url") or d.get("start_url") or None,
        })
    return items


def to_payload(result: dict) -> dict:
    ans = (result.get("answer") or "").strip()
    src = (result.get("source") or "").strip() if ans else ""
    return {"answer": ans, "source": src, "citations": [src] if src else [],
            "confidence": round(float(result.get("confidence") or 0.0), 3)}


def write_output(qid: str, result: dict, out_dir=None, echo=False) -> Path:
    d = Path(out_dir) if out_dir else OUTPUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / OUTPUT_TEMPLATE.format(qid=qid)
    payload = to_payload(result)
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    if echo:
        print(json.dumps(payload, ensure_ascii=False))
    return p
