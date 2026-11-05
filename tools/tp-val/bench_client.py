#!/usr/bin/env python3
"""bench_client.py: fixed-prompt A/B bench against a running Maya OpenAI-compatible server (greedy, temperature 0).

Tests (all deterministic inputs; prompt sizes are cut by the server's own /tokenize + /detokenize):
  warm     one 50-token request (not scored)
  short    3 greedy 400-token decodes on fixed prompts: decode tok/s (server timings) + text + sha
  long     ~25K-token code prompt, then a 300-token greedy decode: prompt tok/s and decode tok/s at that context
  turn     prime a ~10K-token conversation (2 tokens out), then a turn that adds ~2K new tokens and decodes up to
           200: wall seconds (latency), new prompt tokens read, prompt tok/s, decode tok/s

usage: bench_client.py --url http://HOST:PORT --key-file FILE --corpus FILE --out DIR [--tests warm,short,long,turn]
Writes DIR/bench.json (one record per request) and DIR/text-*.txt (reasoning + <<C>> + content)."""
import argparse, hashlib, json, os, sys, time, urllib.request

PROMPTS = ["Explain how a B-tree insert splits a node, step by step.",
           "Write a Python function that merges overlapping intervals, then explain its complexity.",
           "Describe the causes of the French Revolution in a structured overview."]

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--key-file", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tests", default="warm,short,long,turn")
ap.add_argument("--long-tokens", type=int, default=25000)
ap.add_argument("--turn-base-tokens", type=int, default=10000)
ap.add_argument("--turn-new-tokens", type=int, default=2000)
ap.add_argument("--model", default="glm-5.3-flash")
a = ap.parse_args()
KEY = open(os.path.expanduser(a.key_file)).read().strip()
if KEY.startswith("STRATA_API_KEY="):
    KEY = KEY.split("=", 1)[1].strip()
os.makedirs(a.out, exist_ok=True)
CORPUS = open(a.corpus, encoding="utf-8", errors="replace").read()
REC = []


def post(path, body, timeout=1800):
    r = urllib.request.Request(a.url.rstrip("/") + path, json.dumps(body).encode(),
                               {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def cut(offset_chars, n_tokens):
    """Text of exactly n_tokens tokens (as the server tokenizes it) starting at a fixed corpus offset."""
    span = CORPUS[offset_chars:offset_chars + n_tokens * 6]
    toks = post("/tokenize", {"content": span})["tokens"]
    if len(toks) < n_tokens:
        sys.exit(f"corpus too short at offset {offset_chars}: {len(toks)} < {n_tokens} tokens")
    d = post("/detokenize", {"tokens": toks[:n_tokens]})
    return d.get("content") if isinstance(d, dict) else d


def chat(msgs, max_tokens, tag):
    body = {"model": a.model, "temperature": 0, "max_tokens": max_tokens, "messages": msgs}
    t0 = time.time()
    j = post("/v1/chat/completions", body)
    wall = time.time() - t0
    m = j["choices"][0]["message"]
    txt = (m.get("reasoning_content") or "") + "\n<<C>>\n" + (m.get("content") or "")
    t = j.get("timings") or {}
    n = t.get("predicted_n") or (j.get("usage") or {}).get("completion_tokens")
    r = dict(tag=tag, decode_tps=round(t.get("predicted_per_second") or 0, 2), n=n,
             prompt_n=t.get("prompt_n"), prompt_tps=round(t.get("prompt_per_second") or 0, 1),
             prompt_ms=round(t.get("prompt_ms") or 0), cached_n=t.get("cache_n"),
             usage_prompt=(j.get("usage") or {}).get("prompt_tokens"), wall_s=round(wall, 2),
             finish=j["choices"][0].get("finish_reason"), sha=hashlib.sha256(txt.encode()).hexdigest()[:12],
             draft_n=t.get("draft_n"), draft_accepted=t.get("draft_n_accepted"))
    open(os.path.join(a.out, f"text-{tag}.txt"), "w").write(txt)
    REC.append(r)
    print(json.dumps(r), flush=True)
    json.dump(REC, open(os.path.join(a.out, "bench.json"), "w"), indent=1)
    return r


tests = a.tests.split(",")
if "warm" in tests:
    chat([{"role": "user", "content": "Say hello."}], 50, "warm")
if "short" in tests or "short1" in tests:        # short1: only the first prompt (a quick check)
    for i, p in enumerate(PROMPTS if "short" in tests else PROMPTS[:1]):
        chat([{"role": "user", "content": p}], 400, f"short{i}")
if "long" in tests:
    body = cut(0, a.long_tokens)
    chat([{"role": "user", "content": "Review this code:\n" + body +
           "\n\nExplain in detail what the code above does, function by function."}], 300, "long25k")
if "turn" in tests:
    off = 400000                                   # a corpus region the long test does not use
    base = cut(off, a.turn_base_tokens)
    new = cut(off + a.turn_base_tokens * 6, a.turn_new_tokens)
    conv = [{"role": "user", "content": "Review this code:\n" + base}, {"role": "assistant", "content": "OK, reviewed."}]
    chat(conv + [{"role": "user", "content": "Reply OK."}], 2, "turn-prime")
    chat(conv + [{"role": "user", "content": "Now also this part, then summarize the main risk in 3 sentences:\n" + new}],
         200, "turn")
print("done", flush=True)
