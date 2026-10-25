#!/usr/bin/env python3
"""AMD AI League Match 4 -- entrypoint. Plumbing only; the contract lives in io_contract.py."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_contract  # noqa: E402


def main(argv=None):
    args = io_contract.parse_args(argv)
    if args.index is not None:
        import llm
        ok = llm.start_daemon()
        print(f"warm-up done; resident model daemon: {'up' if ok else 'not started (no model dir)'}")
        return 0
    import agent
    import llm
    backend = llm.get_backend()
    items = io_contract.load_batch(args.input) if args.input else [
        {"id": args.query_id, "query": args.query or "", "url": args.url}]
    for it in items:
        t0 = time.time()
        try:
            res = agent.run(it["query"], it["url"], backend=backend)
        except Exception as e:  # never leave the harness without a file
            print(f"[app] error: {e}", file=sys.stderr)
            res = {"answer": "", "source": "", "confidence": 0.0}
        io_contract.write_output(it["id"], res, args.output_dir, echo=args.stdout)
        print(f"[app] {it['id']} {time.time() - t0:.1f}s mode={res.get('mode')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
