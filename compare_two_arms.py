#!/usr/bin/env python3
"""Paired comparison of two planner arms on the same items, rule-based scoring only.

  python compare_two_arms.py <label_base> <label_trained> [tasks...]

Per task: containment (or gameof24 expression check) accuracy for both arms, item flips in each
direction, exact McNemar p, mean tool steps. Labels are the result-dir names under
test/<task>/results/ (run_qa_arm.sh: "<task>-<model basename with / -> _>").
"""
import json, os, sys, re
from math import comb
sys.path.insert(0, "/data/sakhaei/code/AgentFlow")
from score_qa import extract_prediction, contains_all_gold_tokens, normalize
TEST = "/data/sakhaei/code/AgentFlow/test"

def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n)

def g24_ok(pred, question):
    try:
        import score_gameof24 as g
        return int(g.check_answer(pred, question)) if hasattr(g, "check_answer") else None
    except Exception:
        return None

def score_dir(task, label):
    data = json.load(open(f"{TEST}/{task}/data/data.json"))
    out = {}
    g24 = None
    if task == "gameof24":  # score_gameof24.py writes {"summary", "rows":[{pid, correct, ...}]}
        sc = f"{TEST}/{task}/results/{label}/score_rulebased.json"
        if os.path.exists(sc):
            try:
                g24 = {str(r["pid"]): int(bool(r["correct"])) for r in json.load(open(sc))["rows"]}
            except Exception:
                g24 = None
    for d in data:
        pid = str(d.get("pid", d.get("idx")))
        p = f"{TEST}/{task}/results/{label}/output_{pid}.json"
        if not os.path.exists(p): continue
        o = json.load(open(p))
        pred = extract_prediction(o.get("direct_output", ""))
        golds = d["answer"] if isinstance(d["answer"], list) else [d["answer"]]
        if task == "gameof24" and g24 is not None and pid in g24:
            ok = g24[pid]
        else:
            ok = max(contains_all_gold_tokens(pred, str(g)) for g in golds)
        mem = o.get("memory") or {}
        steps = sum(1 for v in mem.values() if isinstance(v, dict))
        out[pid] = (ok, steps)
    return out

def main():
    lb, lt = sys.argv[1], sys.argv[2]
    tasks = sys.argv[3:] or ["bamboogle", "2wiki", "gameof24", "aime24"]
    print(f"{'task':10s} {'n':>4s} {'base':>7s} {'trained':>8s} {'delta':>7s} {'base-only':>9s} {'tr-only':>7s} {'McNemar p':>10s} {'steps b/t':>10s}")
    tb = tt = tn = 0; B = C = 0
    for task in tasks:
        a = score_dir(task, f"{task}-{lb}"); b = score_dir(task, f"{task}-{lt}")
        ks = sorted(set(a) & set(b), key=int)
        if not ks: print(f"{task:10s}  (no paired items)"); continue
        n = len(ks); ab = sum(a[k][0] for k in ks); bb = sum(b[k][0] for k in ks)
        bo = sum(1 for k in ks if a[k][0] and not b[k][0]); to = sum(1 for k in ks if b[k][0] and not a[k][0])
        sa = sum(a[k][1] for k in ks) / n; sb = sum(b[k][1] for k in ks) / n
        print(f"{task:10s} {n:4d} {100*ab/n:6.1f}% {100*bb/n:7.1f}% {100*(bb-ab)/n:+6.1f} {bo:9d} {to:7d} {mcnemar(bo,to):10.3f} {sa:4.2f}/{sb:4.2f}")
        tb += ab; tt += bb; tn += n; B += bo; C += to
    if tn:
        print(f"{'ALL':10s} {tn:4d} {100*tb/tn:6.1f}% {100*tt/tn:7.1f}% {100*(tt-tb)/tn:+6.1f} {B:9d} {C:7d} {mcnemar(B,C):10.3f}")

if __name__ == "__main__":
    main()
