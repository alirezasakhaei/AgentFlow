#!/usr/bin/env python3
"""Build a step sample for ONE generator on ONE task, so the generator column is clean.

  python build_gen.py <task> <label> <out.jsonl> [seed] [per_stratum]

PATCHED (MAReasoning) 2026-09-18 -- gold-label fix. The first version tested
`normalize(gold) in normalize(result)` as a plain substring. On 2wiki roughly half the golds
are "yes"/"no", and "no" matches inside "not", "know", "nominated" ... so a step reporting
"the director is NOT mentioned" was labelled gold. Now:
  * gold must match on word boundaries;
  * items whose gold is yes/no are excluded from gold AND mid (a string test cannot tell
    whether a step answered a yes/no question); their dead steps are kept;
  * comparison items ("who is older ...") are excluded from gold as before, because the
    gold is one of the two names in the question and any echo of the question matches.
Strata: dead = nothing usable came back; gold = result contains the answer; mid = content
but not the answer.
"""
import json, os, re, glob, random, sys
sys.path.insert(0, "/data/sakhaei/code/AgentFlow")
from score_qa import normalize

TEST = "/data/sakhaei/code/AgentFlow/test"
CMP_RE = r"\b(older|younger|taller|shorter|longer|earlier|later|first|more|less)\b"
YESNO = {"yes", "no"}
TASK, LABEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
random.seed(int(sys.argv[4]) if len(sys.argv) > 4 else 4242)
PER = int(sys.argv[5]) if len(sys.argv) > 5 else 10

def result_text(v):
    r = v.get("result")
    if r is None: return ""
    if isinstance(r, list) and r and isinstance(r[0], dict):
        parts = []
        for blob in r:
            rel = blob.get("relevant_pages (to the query)") or blob.get("relevant_pages") or []
            for p in rel: parts.append(f"{p.get('title')}: {p.get('retrieved_information')}")
            if not rel: parts.append(json.dumps(blob)[:400])
        return " || ".join(parts)
    return r if isinstance(r, str) else json.dumps(r)

def gold_in(golds, nres):
    for g in golds:
        ng = normalize(str(g))
        if not ng or ng in YESNO: continue
        if re.search(r"(?<!\w)" + re.escape(ng) + r"(?!\w)", nres): return True
    return False

data = {str(d.get("pid", d.get("idx"))): d for d in json.load(open(f"{TEST}/{TASK}/data/data.json"))}
rows = []; n_yn = n_cmp = 0
for f in glob.glob(f"{TEST}/{TASK}/results/{LABEL}/output_*.json"):
    o = json.load(open(f)); pid = str(o.get("pid"))
    if pid not in data: continue
    item = data[pid]
    q = str(item.get("query") or item.get("question") or "").split("When ready")[0].strip()
    golds = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]
    is_yn = all(normalize(str(g)) in YESNO for g in golds)
    is_cmp = bool(re.search(CMP_RE, q, re.I))
    n_yn += is_yn; n_cmp += is_cmp
    for k, v in (o.get("memory") or {}).items():
        if not isinstance(v, dict): continue
        if "No matched tool" in str(v.get("tool_name")): continue   # harness parse artefact
        res = result_text(v); nres = normalize(res)
        empty = (not res.strip()) or "NO PAGES" in res or '"relevant_pages": []' in res \
                or "Error searching Wikipedia" in res or "No results found for query" in res
        if empty: stratum = "dead"
        elif is_yn: continue                       # cannot label gold/mid by string
        elif gold_in(golds, nres) and not is_cmp: stratum = "gold"
        elif gold_in(golds, nres) and is_cmp: continue   # question echo; unreliable
        else: stratum = "mid"
        rows.append({"task": TASK, "label": LABEL, "pid": pid, "step": k, "question": q,
                     "gold": golds, "tool": str(v.get("tool_name")),
                     "sub_goal": str(v.get("sub_goal") or "")[:700],
                     "command": str(v.get("command") or "")[:500],
                     "result": res[:2000], "stratum": stratum})

by = {"dead": [], "gold": [], "mid": []}
for r in rows: by[r["stratum"]].append(r)
print(f"{TASK}/{LABEL}: items yes/no={n_yn} cmp={n_cmp} | pool {({k: len(v) for k, v in by.items()})}")
sample = []
for s in ("dead", "gold", "mid"):
    random.shuffle(by[s]); sample.extend(by[s][:PER])
random.shuffle(sample)
for i, r in enumerate(sample): r["sheet_id"] = i + 1
with open(OUT, "w") as fh:
    for r in sample: fh.write(json.dumps(r) + "\n")
print(f"  wrote {len(sample)} -> {OUT}  {({s: sum(1 for r in sample if r['stratum']==s) for s in by})}")
