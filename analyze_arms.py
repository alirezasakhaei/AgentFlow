#!/usr/bin/env python3
"""Compare two 2wiki planner arms step by step.

Success rate alone cannot carry this task: a large slice of 2wiki is binary (pick one of two
named films, or yes/no), where guessing scores ~50%. So every rate is reported split by
question shape, and the step-level behaviour is reported alongside -- in particular how often
a step produced NO tool result, which on this harness means the command was silently dropped
by executor.py::split_commands rather than the tool failing.
"""
import json, os, re, sys, collections, statistics as st

sys.path.insert(0, "/data/sakhaei/code/AgentFlow")
from score_qa import normalize, extract_prediction, contains_all_gold_tokens

TEST_ROOT = "/data/sakhaei/code/AgentFlow/test"

def task_of(label):
    """Labels are '<task>-<arm><suffix>'; the pre-generalisation ones were 'arm-<planner>'."""
    head = label.split("-")[0]
    return "2wiki" if head == "arm" else head
CMP_RE = r"\b(older|younger|taller|shorter|longer|earlier|later|first|more|less)\b"
YESNO = ("yes", "no")

def shape(item, task="2wiki"):
    if normalize(str(item["answer"])) in YESNO:
        return "yes/no (binary)"
    # The two-way "X or Y?" template is a 2wiki artefact. On musique/hotpotqa the same
    # comparative words appear in genuinely open questions, so do not label those binary.
    if task == "2wiki" and re.search(CMP_RE, item["query"], re.I):
        return "comparison (binary)"
    return "open"

def analyse(label):
    base = os.path.join(TEST_ROOT, task_of(label))
    res = os.path.join(base, "results", label)
    data = {str(d.get("pid", d.get("idx"))): d
            for d in json.load(open(os.path.join(base, "data", "data.json")))}
    rows, steps_all, tools, first_tool = [], [], collections.Counter(), collections.Counter()
    empty_steps = drop_multi = drop_fstring = nameerr = 0
    total_steps = 0
    for pid, item in sorted(data.items(), key=lambda kv: int(kv[0])):
        f = os.path.join(res, f"output_{pid}.json")
        if not os.path.exists(f):
            continue
        o = json.load(open(f))
        pred = extract_prediction(o.get("direct_output", ""))
        gold = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]
        ok = max(contains_all_gold_tokens(pred, g) for g in gold)
        mem = o.get("memory") or {}
        used = []
        for k, v in mem.items():
            if not isinstance(v, dict):
                continue
            total_steps += 1
            tn = str(v.get("tool_name"))
            tools[tn.split(":")[0][:40]] += 1
            used.append(tn)
            cmd = str(v.get("command") or "")
            r = v.get("result")
            is_empty = (r == [] or r is None or (isinstance(r, list) and not r))
            if is_empty:
                empty_steps += 1
            # why was it empty? these are the two interface mismatches we can detect
            binds = re.findall(r"(\w+)\s*=\s*tool\.execute\(", cmd)
            if len(binds) > 1 or (binds and binds[0] != "execution"):
                drop_multi += 1
            if "f\"" in cmd or "f'" in cmd:
                drop_fstring += 1
            if isinstance(r, str) and "is not defined" in r:
                nameerr += 1
        if used:
            first_tool[used[0].split(":")[0][:40]] += 1
        steps_all.append(o.get("step_count", 0))
        rows.append((pid, shape(item, task_of(label)), ok, o.get("step_count", 0), o.get("execution_time", 0.0), pred))
    return dict(label=label, rows=rows, steps=steps_all, tools=tools, first_tool=first_tool,
                empty=empty_steps, total_steps=total_steps, drop_multi=drop_multi,
                drop_fstring=drop_fstring, nameerr=nameerr)

def report(a):
    rows = a["rows"]
    n = len(rows)
    print(f"\n{'='*78}\nARM: {a['label']}   (n={n})")
    if not n:
        print("  no outputs"); return
    byshape = collections.defaultdict(list)
    for pid, sh, ok, stp, t, pred in rows:
        byshape[sh].append(ok)
    overall = sum(ok for _, _, ok, _, _, _ in rows) / n
    print(f"  SUCCESS RATE (containment): {sum(ok for _,_,ok,_,_,_ in rows)}/{n} = {100*overall:.1f}%")
    for sh in sorted(byshape):
        v = byshape[sh]
        guess = " (guess baseline ~50%)" if "binary" in sh else ""
        print(f"     {sh:22s}: {sum(v):3d}/{len(v):3d} = {100*sum(v)/len(v):5.1f}%{guess}")
    print(f"  mean steps: {st.mean(a['steps']):.2f}   mean s/item: {st.mean([r[4] for r in rows]):.1f}")
    print(f"  steps with NO tool result: {a['empty']}/{a['total_steps']} = {100*a['empty']/max(a['total_steps'],1):.1f}%")
    print(f"     of which commands with multi/non-'execution' bindings : {a['drop_multi']}")
    print(f"     commands using f-strings over cross-step variables    : {a['drop_fstring']}")
    print(f"     explicit NameError results                            : {a['nameerr']}")
    print("  tool calls:")
    for t, c in a["tools"].most_common():
        print(f"     {c:4d}  {t}")
    print("  first tool per item:")
    for t, c in a["first_tool"].most_common():
        print(f"     {c:4d}  {t}")

if __name__ == "__main__":
    arms = sys.argv[1:] or ["arm-agentflow", "arm-base"]
    results = [analyse(x) for x in arms]
    for a in results:
        report(a)
    if len(results) == 2 and all(r["rows"] for r in results):
        print(f"\n{'='*78}\nHEAD-TO-HEAD (items completed by both)")
        m = {r[0]: r for r in results[1]["rows"]}
        both = [(x, m[x[0]]) for x in results[0]["rows"] if x[0] in m]
        a_only = sum(1 for x, y in both if x[2] and not y[2])
        b_only = sum(1 for x, y in both if y[2] and not x[2])
        agree = sum(1 for x, y in both if x[2] == y[2])
        print(f"  paired items: {len(both)}   both same: {agree}")
        print(f"  {results[0]['label']} correct only: {a_only}")
        print(f"  {results[1]['label']} correct only: {b_only}")
