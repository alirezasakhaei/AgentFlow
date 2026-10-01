#!/usr/bin/env python3
"""Judge-scored eval (Qwen3-30B-A3B judge on mds2, AgentFlow's rubric) next to the rule scorer, paired across arms.
  python judge_arms.py <label> [<label> ...] [--tasks bamboogle,2wiki,gameof24,aime24]
Verdicts are cached per results dir in judge_qwen30b.json. First label is the reference for flips.
PATCHED (MAReasoning) 2026-10-01."""
import json, os, sys, argparse
from concurrent.futures import ThreadPoolExecutor
from math import comb
sys.path.insert(0, "/data/sakhaei/code/AgentFlow"); sys.path.insert(0, "/data/sakhaei/code/AgentFlow/train")
from score_qa import extract_prediction, contains_all_gold_tokens
from judge import judge
TEST = "/data/sakhaei/code/AgentFlow/test"
def mcnemar(b, c):
    n = b + c; k = min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
def score_dir(task, label):
    data = {str(d.get("pid", d.get("idx"))): d for d in json.load(open(f"{TEST}/{task}/data/data.json"))}
    rd = f"{TEST}/{task}/results/{task}-{label}"; cache_p = f"{rd}/judge_qwen30b.json"
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    todo = []
    for pid, d in data.items():
        p = f"{rd}/output_{pid}.json"
        if not os.path.exists(p): continue
        o = json.load(open(p)); out = o.get("direct_output", "") or ""
        pred = extract_prediction(out) or out[-1500:]
        golds = d["answer"] if isinstance(d["answer"], list) else [d["answer"]]
        if task == "gameof24":   # gold is ONE valid expression; use the expression checker (any valid expression counts), judge is not meaningful here
            try:
                import score_gameof24 as g24; rule = int(bool(g24.valid_expressions(g24.answer_section(out), d["question"])))
            except Exception: rule = 0
        else:
            rule = max(contains_all_gold_tokens(extract_prediction(out), str(g)) for g in golds)
        if pid not in cache: todo.append((pid, d.get("question") or d.get("query") or "", "; ".join(str(g) for g in golds), pred))
        cache.setdefault(pid, {})["rule"] = rule
    def run(t):
        pid, q, g, a = t; v, raw = judge(q, g, a); return pid, v, raw
    if todo:
        with ThreadPoolExecutor(16) as ex:
            for pid, v, raw in ex.map(run, todo):
                cache[pid].update({"judge": v, "raw": (raw or "")[-200:]})
        json.dump(cache, open(cache_p, "w"), indent=1)
    return {pid: (c.get("judge"), c["rule"]) for pid, c in cache.items() if "judge" in c}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("labels", nargs="+"); ap.add_argument("--tasks", default="bamboogle,2wiki,gameof24,aime24")
    a = ap.parse_args(); tasks = a.tasks.split(",")
    res = {(t, l): score_dir(t, l) for t in tasks for l in a.labels}
    print(f"{'task':9s} {'arm':28s} {'n':>4s} {'judge':>6s} {'rule':>6s} {'agree':>6s} {'judge-only':>10s}  flips vs ref (judge): ref-only/arm-only p")
    for t in tasks:
        ref = res[(t, a.labels[0])]
        for l in a.labels:
            r = res[(t, l)]; ks = sorted(set(r) & set(ref), key=int)
            if not r: continue
            n = len(r); J = sum(1 for v in r.values() if v[0]); R = sum(v[1] for v in r.values())
            ag = sum(1 for v in r.values() if bool(v[0]) == bool(v[1])); jo = sum(1 for v in r.values() if v[0] and not v[1])
            fl = ""
            if l != a.labels[0] and ks:
                bo = sum(1 for k in ks if ref[k][0] and not r[k][0]); to = sum(1 for k in ks if r[k][0] and not ref[k][0])
                fl = f"  {bo}/{to} p={mcnemar(bo, to):.3f} (n={len(ks)})"
            print(f"{t:9s} {l:28s} {n:4d} {100*J/n:5.1f}% {100*R/n:5.1f}% {100*ag/n:5.1f}% {100*jo/n:9.1f}%{fl}")
    print("ALL (judge / rule), paired on common items:")
    for l in a.labels:
        tot = {}; [tot.update({(t, k): v for k, v in res[(t, l)].items()}) for t in tasks]
        ref = {}; [ref.update({(t, k): v for k, v in res[(t, a.labels[0])].items()}) for t in tasks]
        ks = set(tot) & set(ref); n = len(ks)
        if not n: continue
        J = sum(1 for k in ks if tot[k][0]); R = sum(tot[k][1] for k in ks)
        bo = sum(1 for k in ks if ref[k][0] and not tot[k][0]); to = sum(1 for k in ks if tot[k][0] and not ref[k][0])
        print(f"  {l:28s} n={n:3d} judge {100*J/n:5.1f}%  rule {100*R/n:5.1f}%" + ("" if l == a.labels[0] else f"  judge flips ref-only/arm-only {bo}/{to} p={mcnemar(bo,to):.3f}"))
if __name__ == "__main__": main()
