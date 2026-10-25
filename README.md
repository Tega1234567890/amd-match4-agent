# AMD AI League – Match 4 "Intelligent Web Information Retrieval" agent

An agent that takes a natural-language question (optionally with a start URL), browses, and returns **one exact value plus
the source URL**, or an empty answer if it can't find a grounded one.

> **The contract below is ASSUMED and UNOFFICIAL.** As of 25 Oct 2026, AMD/lablab had not published the Match 4 I/O
> spec (CLI, JSON keys, base image, limits). It is modelled on the Mini-Challenge 3 starter kit (container, one exec per
> question, JSON file out) and on public Match 4 project pages. All of it is in `app/io_contract.py` so it can be
> changed in one place once the official spec lands.

## Image
Image reference is provided privately in the lablab submission. The image is built by `.github/workflows/build-image.yml` on GitHub Actions:
- base `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0` (the MC2/MC3 mandated base, not squashed)
- Qwen2.5-1.5B-Instruct weights baked in at `/models/llm`, so the model loads offline
- code in `/app`, `CMD sleep infinity`, no port

## Assumed contract
- **Warm-up, once:** `python3 /app/app.py --index` loads the model once into a resident unix-socket daemon, so the
  per-question processes don't reload it. An optional corpus argument is accepted and ignored.
- **Per question:** `python3 /app/app.py --query-id query_01 --query "natural-language question" [--url https://start.page]`
  (aliases `--question`, `--id`, `--start-url`; a URL inside the question text is also used).
- **Batch:** `python3 /app/app.py --input questions.json` (a list, or `{"queries": [...]}`, of `{id, query|question, url|start_url}`).
- **Output** (always written, with an empty answer on any error): `/app/output/<query-id>_output.json`
  ```json
  {"answer": "1889", "source": "https://...", "citations": ["https://..."], "confidence": 0.8}
  ```
  `answer` is the value only, never prose. `""` with an empty source/citations means not found.

## How it works
1. **Seeds:** the start URL if given, otherwise web search (DuckDuckGo HTML, falling back to the Wikipedia API).
2. **Crawl:** best-first, same-site, bounded (6 pages, 25 s deadline); links ranked by BM25 against the question.
3. **Parse:** structure-aware text, with table and infobox rows turned into `label: value` lines.
4. **Rank:** BM25 chunk ranking with an answer-type bonus (year, date, number, money, version).
5. **Extract:** an LLM picks the exact span. Backends: resident daemon > local transformers model on the GPU (ROCm shows up
   as `cuda`) > OpenAI-compatible endpoint (dev only) > typed-regex heuristic.
6. **Ground:** the answer must appear on the cited page, otherwise refuse.

## Run
```
docker run -d --name m4 <image>   # image reference provided privately in the lablab submission
docker exec m4 python3 /app/app.py --index
docker exec m4 python3 /app/app.py --query-id q1 --query "Which TCP port does HTTPS use by default?" \
       --url https://en.wikipedia.org/wiki/HTTPS --stdout
```
Locally without Docker: `pip install -r app/requirements.txt torch`, then
`hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/llm`, `export M4_MODEL_DIR=$PWD/models/llm`, and run `python app/app.py ...`.
To run the test set: `python tests/run_tests.py`.
Env knobs: `M4_DEADLINE_S` (25), `M4_MAX_PAGES` (6), `M4_NO_LLM=1`, `M4_OUTPUT_DIR`, `M4_OUTPUT_TEMPLATE`, `M4_MODEL_DIR`.

## Local test result (25 Oct 2026, CPU only)
**9/10** on 10 public factual questions (max 12.3 s per question). Heuristic-only mode scores 3/10. See `tests/RESULTS.md`.
These are easy, mostly Wikipedia questions, so treat the result as a smoke test.

## To tighten once the official spec is published
1. **`io_contract.py`:** the CLI flags, the input format, the output path and filename, the JSON keys and the refusal format.
2. **Network at grading time:** MC3 was graded offline, but a web agent needs to browse. Confirm outbound access, or
   whether a search API or proxy is provided. DuckDuckGo is often bot-walled from datacenter IPs.
3. **Base image and limits:** the base tag, image size, VRAM band and time limits. A larger model, such as Qwen2.5-7B, is an option if the
   per-question budget allows it on the grading GPU.
4. **Answer normalisation:** units and date formats, to match the grader.
5. **JS-rendered pages:** there's no headless browser yet.
6. **Known weaknesses:** wrong-entity evidence on search-only questions, and grounding is checked per page rather than per chunk.
