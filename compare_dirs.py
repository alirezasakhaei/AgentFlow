#!/usr/bin/env python3
"""Paired comparison of two result DIRS (full names) for one task, same scoring as compare_two_arms.py."""
import json, os, sys
sys.path.insert(0, "/data/sakhaei/code/AgentFlow")
from compare_two_arms import mcnemar
from score_qa import extract_prediction, contains_all_gold_tokens
TEST = "/data/sakhaei/code/AgentFlow/test"
def score(task, d):
    data = json.load(open(f"{TEST}/{task}/data/data.json")); out = {}
    for it in data:
        pid = str(it.get("pid", it.get("idx"))); p = f"{TEST}/{task}/results/{d}/output_{pid}.json"
        if not os.path.exists(p): continue
        o = json.load(open(p)); txt = o.get("direct_output", "") or ""
        if task == "gameof24":
            import score_gameof24 as g; ok = int(bool(g.valid_expressions(g.answer_section(txt), it["question"])))
        else:
            golds = it["answer"] if isinstance(it["answer"], list) else [it["answer"]]
            ok = max(contains_all_gold_tokens(extract_prediction(txt), str(x)) for x in golds)
        mem = o.get("memory") or {}; out[pid] = (ok, sum(1 for v in mem.values() if isinstance(v, dict)))
    return out
task, A, B = sys.argv[1], sys.argv[2], sys.argv[3]
a, b = score(task, A), score(task, B); ks = sorted(set(a) & set(b), key=int); n = len(ks)
ab = sum(a[k][0] for k in ks); bb = sum(b[k][0] for k in ks)
bo = sum(1 for k in ks if a[k][0] and not b[k][0]); to = sum(1 for k in ks if b[k][0] and not a[k][0])
print(f"{task:10s} n={n:3d}  A={100*ab/n:5.1f}%  B={100*bb/n:5.1f}%  delta={100*(bb-ab)/n:+5.1f}  A-only={bo} B-only={to}  p={mcnemar(bo,to):.3f}  steps {sum(a[k][1] for k in ks)/n:.2f}/{sum(b[k][1] for k in ks)/n:.2f}   [A={A} B={B}]")
