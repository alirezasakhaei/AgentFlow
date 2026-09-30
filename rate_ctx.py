#!/usr/bin/env python3
"""Rate logged steps under four context conditions.

  python rate_ctx.py <sample.jsonl> <out.jsonl> <c0|c1|c2|c3> [rater ...]

  c0  the step alone (the original condition)
  c1  + every step that PRECEDED it                     -- causal history
  c2  + the WHOLE trajectory, step under review marked  -- hindsight
  c3  + the whole trajectory AND the gold answer        -- oracle ceiling

The rating question is byte-identical in all four; only the context varies, so the
comparison isolates context rather than confounding it with framing. Ratings are stored
per condition under ratings[rater][cond], so conditions merge into one file.
"""
import json, os, re, sys, urllib.request, concurrent.futures as cf

ALL = {
    "agentflow-7b": ("http://localhost:8000/v1/chat/completions", "AgentFlow/agentflow-planner-7b"),
    "qwen-base-7b": ("http://localhost:8001/v1/chat/completions", "Qwen/Qwen2.5-7B-Instruct"),
    "qwen-32b-awq": ("http://mds1srv2.epfl.ch:8000/v1/chat/completions", "Qwen/Qwen2.5-32B-Instruct-AWQ"),
    "qwen-72b-awq": ("http://mds1srv2.epfl.ch:8000/v1/chat/completions", "Qwen/Qwen2.5-72B-Instruct-AWQ"),
}
TEST = "/data/sakhaei/code/AgentFlow/test"
QUESTION_BLOCK = """How much information did that step give you toward answering the original question?
Reply with a single integer from 0 to 4 on this scale:
0 = nothing at all (empty, an error, or no usable content)
1 = almost nothing
2 = some partial or indirect information
3 = substantial, directly relevant information
4 = it essentially answers the question

Reply with only the integer and nothing else."""

def flat_result(v, cap=600):
    r = v.get("result")
    if r is None: return ""
    if isinstance(r, list) and r and isinstance(r[0], dict):
        parts = []
        for blob in r:
            rel = blob.get("relevant_pages (to the query)") or blob.get("relevant_pages") or []
            for p in rel: parts.append(f"{p.get('title')}: {p.get('retrieved_information')}")
            if not rel: parts.append(json.dumps(blob)[:300])
        s = " || ".join(parts)
    else:
        s = r if isinstance(r, str) else json.dumps(r)
    return s[:cap]

def step_block(k, v, cap=600):
    return (f"[{k}]\n  sub-goal: {str(v.get('sub_goal') or '')[:300]}\n"
            f"  tool: {v.get('tool_name')}\n"
            f"  command: {str(v.get('command') or '')[:200]}\n"
            f"  result: {flat_result(v, cap) or '(nothing returned)'}")

def trajectory(row):
    """Returns (ordered list of (key, memdict), index of the step under review)."""
    path = f"{TEST}/{row['task']}/results/{row['label']}/output_{row['pid']}.json"
    mem = (json.load(open(path)).get("memory") or {})
    items = [(k, v) for k, v in mem.items() if isinstance(v, dict)]
    items.sort(key=lambda kv: int(re.findall(r"\d+", kv[0])[0]) if re.findall(r"\d+", kv[0]) else 0)
    idx = next((i for i, (k, _) in enumerate(items) if k == row["step"]), None)
    return items, idx

def build(row, cond):
    q = row["question"][:400]
    under = (f"The step's stated sub-goal: {row['sub_goal'][:400]}\n"
             f"Tool called: {row['tool'][:60]}\n"
             f"Command issued: {row['command'][:300]}\n"
             f"What came back: {row['result'][:1200] or '(nothing returned)'}")
    if cond == "c0":
        return (f"You are reviewing a single step taken by a tool-using agent.\n\n"
                f"Original question: {q}\n\n{under}\n\n{QUESTION_BLOCK}")

    items, idx = trajectory(row)
    if idx is None:
        items, idx = [], None

    if cond == "c1":
        prior = "\n".join(step_block(k, v) for k, v in items[:idx]) if idx else ""
        return (f"You are reviewing one step taken by a tool-using agent, together with "
                f"everything the agent had done before it.\n\n"
                f"Original question: {q}\n\n"
                f"=== STEPS THE AGENT TOOK BEFORE THIS ONE ===\n"
                f"{prior or '(none - this was the first step)'}\n\n"
                f"=== THE STEP UNDER REVIEW ===\n{under}\n\n{QUESTION_BLOCK}")

    full = "\n".join(
        ("*** THE STEP UNDER REVIEW ***\n" if i == idx else "") + step_block(k, v)
        for i, (k, v) in enumerate(items))
    head = (f"You are reviewing one step taken by a tool-using agent, together with the "
            f"agent's complete trajectory for this question.\n\n"
            f"Original question: {q}\n")
    if cond == "c3":
        head += f"The correct answer to the question is: {' | '.join(map(str, row['gold']))}\n"
    return (f"{head}\n=== THE AGENT'S COMPLETE TRAJECTORY ===\n{full}\n\n"
            f"=== THE STEP UNDER REVIEW (marked above) ===\n{under}\n\n{QUESTION_BLOCK}")

def post(url, body, timeout=300):
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
            return model in [d["id"] for d in json.loads(r.read()).get("data", [])]
    except Exception:
        return False

sample_path, out_path, cond = sys.argv[1], sys.argv[2], sys.argv[3]
assert cond in ("c0", "c1", "c2", "c3")
want = sys.argv[4:] or list(ALL)
live = [m for m in want if alive(m)]
print(f"condition {cond} | raters live: {live}" + (f" | skipped: {[m for m in want if m not in live]}" if len(live) < len(want) else ""))
if not live: sys.exit("no raters reachable")

rows = [json.loads(l) for l in open(sample_path)]
prev = {}
if os.path.exists(out_path):
    for l in open(out_path):
        r = json.loads(l); prev[r["sheet_id"]] = r.get("ctx_ratings", {})

def work(job):
    row, name = job
    url, model = ALL[name]
    try:
        p = build(row, cond)
        raw = post(url, {"model": model, "messages": [{"role": "user", "content": p}],
                         "temperature": 0.0, "top_p": 1.0, "max_tokens": 8})
        txt = raw["choices"][0]["message"]["content"]
        m = re.search(r"[0-4]", str(txt))
        return row["sheet_id"], name, (int(m.group()) if m else None), len(p)
    except Exception as e:
        return row["sheet_id"], name, None, f"ERR {type(e).__name__}"

acc, plens = {}, []
with cf.ThreadPoolExecutor(max_workers=6) as ex:
    for sid, name, val, meta in ex.map(work, [(r, n) for r in rows for n in live]):
        acc.setdefault(sid, {})[name] = val
        if isinstance(meta, int): plens.append(meta)

with open(out_path, "w") as fh:
    for r in rows:
        cr = dict(prev.get(r["sheet_id"], {}))
        for name, val in acc.get(r["sheet_id"], {}).items():
            cr.setdefault(name, {})[cond] = val
        r["ctx_ratings"] = cr
        fh.write(json.dumps(r) + "\n")
print(f"wrote -> {out_path}   (mean prompt {sum(plens)//max(len(plens),1)} chars)")

import statistics as st
for name in live:
    print(f"\n{name} [{cond}]")
    for s in ("dead", "gold", "mid"):
        sv = [acc[r["sheet_id"]][name] for r in rows
              if r["stratum"] == s and acc.get(r["sheet_id"], {}).get(name) is not None]
        if sv: print(f"   {s:5s} n={len(sv):2d} mean={st.mean(sv):.2f}")
