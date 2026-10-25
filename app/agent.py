"""Match 4 retrieval core: question (+ optional start URL) -> exact value + source URL.

Pipeline (adapted from web_agent.py v0.1, 20 Oct 2026):
  1. pick seed pages: the start URL if given, else web search (DuckDuckGo HTML -> Wikipedia API fallback)
  2. best-first bounded crawl, same host, links ranked by BM25 against the question
  3. structure-aware text extraction (table/infobox rows become "label: value" lines)
  4. BM25 chunk ranking + answer-type bonus
  5. LLM picks the exact span (local GPU model / daemon / OpenAI-compatible), else a typed regex heuristic
  6. verify the answer occurs verbatim on the cited page; refuse ("") if no grounded answer
Everything is bounded by a wall-clock deadline (default 25 s) so one question never blows the 30 s budget.
"""
from __future__ import annotations

import collections
import math
import os
import re
import sys
import time
import unicodedata
import urllib.parse

import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": os.environ.get("M4_UA", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 M4RetrievalAgent/0.2"),
      "Accept-Language": "en-US,en;q=0.9"}
STOP = set("the a an of to and or in on for is are was were be by with at as from that this it what which who whom "
           "when where how does did do has have had its their his her many much name named called list give tell "
           "according page website site current currently".split())
BLOCK_TAGS = ["p", "li", "h1", "h2", "h3", "h4", "dt", "dd", "caption", "blockquote", "pre", "figcaption"]


def log(*a):
    print("[agent]", *a, file=sys.stderr)


def toks(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return [w for w in re.findall(r"[a-z0-9]+", s) if w not in STOP]


# ---------------------------------------------------------------- fetching / parsing
def fetch(url, timeout=8):
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    if "html" not in ctype and "xml" not in ctype:
        return [r.text[:20000]] if ctype.startswith("text") else [], [], url
    soup = BeautifulSoup(r.text, "lxml")
    for t in soup(["script", "style", "noscript", "svg", "nav", "footer", "form", "iframe"]):
        t.decompose()
    for t in soup.select("sup.reference, .mw-editsection, .navbox, .reflist, #toc"):
        t.decompose()
    links = []
    for a in soup.find_all("a", href=True):
        href = urllib.parse.urljoin(r.url, a["href"]).split("#")[0]
        if href.startswith("http"):
            links.append((href, a.get_text(" ", strip=True)))
    blocks = []
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    if title:
        blocks.append(f"Page title: {title}")
    # tables -> "label: value" rows (infoboxes, spec tables)
    for table in soup.find_all("table"):
        cap = table.find("caption")
        cap = cap.get_text(" ", strip=True) if cap else ""
        headers = [th.get_text(" ", strip=True) for th in table.select("tr th")][:12]
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            cells = [re.sub(r"\s+", " ", c) for c in cells if c]
            if len(cells) >= 2:
                blocks.append((cap + " | " if cap else "") + cells[0] + ": " + " | ".join(cells[1:]))
            elif len(cells) == 1 and len(cells[0]) < 300:
                blocks.append(cells[0])
        table.decompose()
    for el in soup.find_all(BLOCK_TAGS):
        if el.find(BLOCK_TAGS):
            continue
        t = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if len(t) > 2:
            blocks.append(t)
    if len(blocks) < 3:  # div-only pages
        blocks += [b for b in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", soup.get_text(" ", strip=True))) if b]
    return blocks, links, r.url


def chunk_blocks(blocks, max_words=70):
    out, cur, n = [], [], 0
    for b in blocks:
        w = len(b.split())
        if w > max_words:  # long paragraph -> sentence windows
            if cur:
                out.append(" ".join(cur)); cur, n = [], 0
            sents = re.split(r"(?<=[.!?])\s+", b)
            buf = []
            for s in sents:
                buf.append(s)
                if len(" ".join(buf).split()) >= max_words:
                    out.append(" ".join(buf)); buf = buf[-1:]
            if buf:
                out.append(" ".join(buf))
            continue
        if n + w > max_words and cur:
            out.append(" ".join(cur)); cur, n = [], 0
        cur.append(b if b.endswith((".", ":", "?", "!")) else b + " ;"); n += w
    if cur:
        out.append(" ".join(cur))
    return out


def bm25(query, docs, k1=1.5, b=0.75):
    q = toks(query); D = [toks(d) for d in docs]; N = len(D) or 1
    avg = (sum(map(len, D)) / N) or 1
    df = collections.Counter(t for d in D for t in set(d))
    out = []
    for d in D:
        tf = collections.Counter(d); s = 0.0
        for t in q:
            if t in tf:
                idf = math.log(1 + (N - df[t] + .5) / (df[t] + .5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        out.append(s)
    return out


# ---------------------------------------------------------------- search (only when no start URL)
def search(query, n=4, timeout=6):
    urls = []
    try:  # DuckDuckGo HTML endpoint (keyless). Often bot-walled from datacenter IPs.
        r = requests.post("https://html.duckduckgo.com/html/", data={"q": query}, headers=UA, timeout=timeout)
        soup = BeautifulSoup(r.text, "lxml")
        for a in soup.select("a.result__a"):
            h = a.get("href", "")
            if "uddg=" in h:
                h = urllib.parse.unquote(urllib.parse.parse_qs(urllib.parse.urlparse(h).query)["uddg"][0])
            if h.startswith("http") and "duckduckgo.com" not in h:
                urls.append(h)
    except Exception as e:
        log("ddg failed", e)
    if not urls:  # Wikipedia full-text search fallback (always reachable, no key)
        try:
            r = requests.get("https://en.wikipedia.org/w/api.php", headers=UA, timeout=timeout,
                             params={"action": "query", "list": "search", "srsearch": query, "srlimit": n,
                                     "format": "json"})
            for hit in r.json()["query"]["search"]:
                urls.append("https://en.wikipedia.org/wiki/" + urllib.parse.quote(hit["title"].replace(" ", "_")))
        except Exception as e:
            log("wikipedia search failed", e)
    return urls[:n]


def search_query(question):
    q = re.sub(r"https?://\S+", "", question)
    words = [w for w in re.findall(r"[A-Za-z0-9'\-\.]+", q) if w.lower() not in STOP]
    return " ".join(words)[:200]


# ---------------------------------------------------------------- answer typing / heuristics
MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?"
TYPE_PATTERNS = {
    "year": r"\b(1[0-9]{3}|20[0-9]{2})\b",
    "date": rf"\b(\d{{1,2}} {MONTHS} \d{{4}}|{MONTHS} \d{{1,2}},? \d{{4}})\b",
    "number": r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\w])",
    "money": r"([$€£₦]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|k))?)",
    "version": r"\b(v?\d+\.\d+(?:\.\d+)*)\b",
}


def answer_type(q):
    ql = q.lower()
    if re.search(r"\bwhat year\b|\bwhich year\b|\bin what year\b|\byear\b.*\b(founded|built|born|established|released)", ql):
        return "year"
    if re.search(r"\bwhen\b|\bwhat date\b|\bon what date\b", ql):
        return "date"
    if re.search(r"\bversion\b", ql):
        return "version"
    if re.search(r"\bhow much\b|\bprice\b|\bcost\b|\bprize\b|\bfee\b", ql):
        return "money"
    if re.search(r"\bhow many\b|\bhow (tall|high|long|far|deep|big|old)\b|\bpopulation\b|\bnumber of\b|\bheight\b|\barea\b", ql):
        return "number"
    return "text"


def heuristic(question, ranked):
    """No-LLM fallback: typed regex on the best chunks, nearest to query terms."""
    typ = answer_type(question)
    pats = [TYPE_PATTERNS[typ]] if typ in TYPE_PATTERNS else []
    if typ == "date":
        pats.append(TYPE_PATTERNS["year"])
    qt = set(toks(question))
    for _, url, chunk in ranked[:4]:
        for pat in pats:
            best = None
            for m in re.finditer(pat, chunk, flags=re.I):
                window = toks(chunk[max(0, m.start() - 120):m.end() + 60])
                score = len(qt & set(window))
                if best is None or score > best[0]:
                    best = (score, m.group(1))
            if best and best[0] > 0:
                return best[1], url, 0.35
    return "", "", 0.0


def norm(s):
    s = unicodedata.normalize("NFKC", s).upper()
    return re.sub(r"[\s\-\.·_,]", "", s)


def grounded(ans, text):
    if not ans:
        return False
    if norm(ans) in norm(text):
        return True
    at = toks(ans)
    return bool(at) and sum(t in set(toks(text)) for t in at) / len(at) >= 0.75


def clean_answer(ans):
    ans = ans.strip().strip('"').strip("'").strip()
    ans = re.sub(r"^(the answer is|answer:)\s*", "", ans, flags=re.I)
    ans = ans.rstrip(".;,")
    return ans


# ---------------------------------------------------------------- main entry
def run(question, start_url=None, backend=None, deadline_s=None, max_pages=None):
    t0 = time.time()
    deadline = t0 + float(deadline_s or os.environ.get("M4_DEADLINE_S", "25"))
    max_pages = int(max_pages or os.environ.get("M4_MAX_PAGES", "6"))
    if not start_url:
        m = re.search(r"https?://\S+", question)
        start_url = m.group(0).rstrip(").,") if m else None
    seeds = [start_url] if start_url else search(search_query(question))
    host_ok = {urllib.parse.urlparse(u).netloc for u in seeds}
    frontier = [(1e9 - i, u, 0) for i, u in enumerate(seeds)]
    seen, pool, page_text = set(), [], {}
    while frontier and len(seen) < max_pages and time.time() < deadline - 10:
        frontier.sort(reverse=True)
        _, url, depth = frontier.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            blocks, links, final = fetch(url, timeout=min(8, max(2, deadline - 10 - time.time())))
        except Exception as e:
            log("skip", url, e); continue
        page_text[url] = " ".join(blocks)
        cs = chunk_blocks(blocks)
        sc = bm25(question, cs)
        typ = answer_type(question)
        for s, c in zip(sc, cs):
            if typ in TYPE_PATTERNS and re.search(TYPE_PATTERNS[typ], c, re.I):
                s *= 1.25
            pool.append((s, url, c))
        if depth < 2:
            cand = [(h, t) for h, t in links if urllib.parse.urlparse(h).netloc in host_ok and h not in seen
                    and not re.search(r"\.(pdf|jpg|png|zip|svg)$|/(Special|File|Help|Talk|Category):|action=", h)]
            if cand:
                ls = bm25(question, [t + " " + h.replace("_", " ").replace("-", " ") for h, t in cand])
                best_page = max((s for s, u, _ in pool if u == url), default=0)
                frontier += [(s - best_page * 0.5, h, depth + 1) for s, (h, _) in zip(ls, cand) if s > 0][:40]
        # early stop: a strong chunk already found on the start page
        if start_url and len(seen) >= 2 and pool and max(p[0] for p in pool) > 8:
            break
    pool.sort(key=lambda x: -x[0])
    ranked = pool[:6]
    result = {"question": question, "answer": "", "source": "", "confidence": 0.0,
              "pages_visited": sorted(seen), "mode": "none"}
    if not ranked:
        return result
    if backend is not None and time.time() < deadline - 2:
        try:
            from llm import build_messages, parse_reply
            ans, idx = parse_reply(backend.complete(build_messages(question, [(u, c) for _, u, c in ranked])))
            ans = clean_answer(ans)
            src = ranked[idx - 1][1] if 1 <= idx <= len(ranked) else ranked[0][1]
            if ans and not grounded(ans, page_text.get(src, "")):
                hit = next((u for u, t in page_text.items() if grounded(ans, t)), None)
                src = hit or ""
            if ans and src:
                result.update(answer=ans, source=src, confidence=0.8, mode=backend.name)
                return result
            result["mode"] = backend.name + "(refused/ungrounded)"
        except Exception as e:
            log("llm failed", e)
    ans, src, conf = heuristic(question, ranked)
    if ans:
        result.update(answer=ans, source=src, confidence=conf, mode="heuristic")
    result["elapsed_s"] = round(time.time() - t0, 2)
    return result
