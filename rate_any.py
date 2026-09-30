#!/usr/bin/env python3
"""Rate a step sample with whichever raters are currently reachable.

  python rate_any.py <sample.jsonl> <out.jsonl> [rater ...]

Raters not reachable are skipped with a note rather than failing the run, so the grid can be
filled across GPU swaps: the two 7Bs live on mds1 permanently, while the 32B and 72B
time-share mds2's single H100.
Existing ratings in <out.jsonl> are preserved and merged, so a later pass can add a rater.
"""
import json, os, re, sys, urllib.request, concurrent.futures as cf

ALL = {
    "agentflow-7b": ("http://localhost:8000/v1/chat/completions", "AgentFlow/agentflow-planner-7b"),
    "qwen-base-7b": ("http://localhost:8001/v1/chat/completions", "Qwen/Qwen2.5-7B-Instruct"),
    "qwen-32b-awq": ("http://mds1srv2.epfl.ch:8000/v1/chat/completions", "Qwen/Qwen2.5-32B-Instruct-AWQ"),
    "qwen-72b-awq": ("http://mds1srv2.epfl.ch:8000/v1/chat/completions", "Qwen/Qwen2.5-72B-Instruct-AWQ"),
}
PROMPT = """You are reviewing a single step taken by a tool-using agent.

Original question: {question}

The step's stated sub-goal: {sub_goal}
Tool called: {tool}
Command issued: {command}
What came back: {result}

How much information did that step give you toward answering the original question?
Reply with a single integer from 0 to 4 on this scale:
0 = nothing at all (empty, an error, or no usable content)
1 = almost nothing
2 = some partial or indirect information
3 = substantial, directly relevant information
4 = it essentially answers the question

Reply with only the integer and nothing else."""

def post(url, body, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy-token"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def alive(name):
    url, model = ALL[name]
    base = url.rsplit("/chat/completions", 1)[0] + "/models"
    try:
        req = urllib.request.Request(base, headers={"Authorization": "Bearer dummy-token"})
        with urllib.request.urlopen(req, timeout=10) as r:
            ids = [d["id"] for d in json.loads(r.read()).get("data", [])]
        return model in ids
    except Exception:
        return False

sample_path, out_path = sys.argv[1], sys.argv[2]
want = sys.argv[3:] or list(ALL)
live = [m for m in want if alive(m)]
skipped = [m for m in want if m not in live]
print(f"raters live: {live}")
if skipped:
    print(f"raters skipped (not serving): {skipped}")
if not live:
    sys.exit("no raters reachable")

rows = [json.loads(l) for l in open(sample_path)]
prev = {}
if os.path.exists(out_path):
    for l in open(out_path):
        r = json.loads(l)
        prev[r["sheet_id"]] = r.get("ratings", {})

def work(job):
    row, name = job
    url, model = ALL[name]
    p = PROMPT.format(question=row["question"][:400], sub_goal=row["sub_goal"][:400],
                      tool=row["tool"][:60], command=row["command"][:300],
                      result=(row["result"][:1200] or "(nothing returned)"))
    try:
        raw = post(url, {"model": model, "messages": [{"role": "user", "content": p}],
                         "temperature": 0.0, "top_p": 1.0, "max_tokens": 8})
        txt = raw["choices"][0]["message"]["content"]
        m = re.search(r"[0-4]", str(txt))
        return row["sheet_id"], name, (int(m.group()) if m else None), str(txt)[:40]
    except Exception as e:
        return row["sheet_id"], name, None, f"ERR {type(e).__name__}"

acc = {}
with cf.ThreadPoolExecutor(max_workers=6) as ex:
    for sid, name, val, raw in ex.map(work, [(r, n) for r in rows for n in live]):
        acc.setdefault(sid, {})[name] = {"rating": val, "raw": raw}

with open(out_path, "w") as fh:
    for r in rows:
        merged = dict(prev.get(r["sheet_id"], {}))
        merged.update(acc.get(r["sheet_id"], {}))
        r["ratings"] = merged
        fh.write(json.dumps(r) + "\n")
print(f"wrote -> {out_path}")

import collections, statistics as st
for name in live:
    vals = [r["ratings"].get(name, {}).get("rating") for r in rows]
    ok = [v for v in vals if v is not None]
    print(f"\n{name}: parsed {len(ok)}/{len(vals)}  dist {dict(sorted(collections.Counter(ok).items()))}")
    for s in ("dead", "gold", "mid"):
        sv = [r["ratings"].get(name, {}).get("rating") for r in rows if r["stratum"] == s]
        sv = [v for v in sv if v is not None]
        if sv:
            print(f"   {s:5s} n={len(sv):2d} mean={st.mean(sv):.2f}")
