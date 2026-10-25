# Test run - 25 Oct 2026, ~10:55 WAT (box, CPU only, no GPU)

Setup: Qwen2.5-1.5B-Instruct (fp32 on CPU) kept resident by `app.py --index` (unix-socket daemon). Each question runs
as a NEW process via `tests/run_tests.py` (one process per question, like the MC3 harness). Scoring is exact match after
MC-style normalisation (uppercase; strip whitespace and - . · _ , ; units m/°C dropped) against expected answer + aliases.

| id | question | url given | expected | got | result | time |
|---|---|---|---|---|---|---|
| q01 | Eiffel Tower completed (year) | yes (wiki) | 1889 | 1889 | PASS | 9.9 s |
| q02 | Capital of Nigeria | yes (wiki) | Abuja | Abuja | PASS | 9.1 s |
| q03 | Creator of Python | no (search) | Guido van Rossum | Jeff Bezanson | **FAIL** | 7.2 s |
| q04 | Atomic number of gold | no (search) | 79 | 79 | PASS | 7.5 s |
| q05 | Kilimanjaro height (m) | yes (wiki) | 5895 | 5,895 | PASS | 7.0 s |
| q06 | Nigeria independence date | no (search) | 1 October 1960 | 1 October 1960 | PASS | 12.3 s |
| q07 | Default HTTPS port | yes (wiki) | 443 | 443 | PASS | 5.1 s |
| q08 | HTTPStatus.IM_A_TEAPOT value | yes (docs.python.org index; must crawl 1 hop) | 418 | 418 (from library/http.html) | PASS | 5.6 s |
| q09 | Author of Things Fall Apart | no (search) | Chinua Achebe | Chinua Achebe | PASS | 6.5 s |
| q10 | Refusal: TQ-40 junction temp on Eiffel page | yes | "" (empty) | "" | PASS | 5.1 s |

**Score: 9/10 with the LLM.** Max latency 12.3 s on CPU (budget 30 s); a GPU will be much faster.
Baseline with no LLM (`M4_NO_LLM=1`, typed-regex heuristic only): **3/10**.

Failure analysis
- q03: no start URL, so the agent used the Wikipedia search fallback (DuckDuckGo is bot-walled from this box's IP).
  Search pulled in other language pages; the 1.5B model picked "Jeff Bezanson" (Julia's creator) from an evidence block
  and it passed the grounding check because that name appears on a crawled page. Fixes: a bigger model on the GPU,
  entity-overlap filtering of evidence (question entity "Python" must be in the chunk/page title), and a stricter
  check that the answer sits in the SAME chunk the model cited.
- Heuristic-only mode fails on years and heights (it picks the wrong number in the chunk), so the LLM is required.

Caveat: 10 easy, mostly Wikipedia questions. AMD's hidden set is likely harder (non-Wikipedia sites, JS-rendered pages,
multi-hop navigation, tables/PDFs). Treat 9/10 as a smoke test, not a score forecast.
