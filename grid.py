#!/usr/bin/env python3
"""The full rater x generator grid for self-rated step informativeness."""
import json, statistics as st
RATERS = [("agentflow-7b", "AgentFlow-7B (trained)"),
          ("qwen-base-7b", "Qwen 7B base"),
          ("qwen-32b-awq", "Qwen 32B AWQ"),
          ("qwen-72b-awq", "Qwen 72B AWQ")]
GENS = [("grid_7bsteps.jsonl",  "7B-generated"),
        ("grid_32bsteps.jsonl", "32B-generated"),
        ("grid_72bsteps.jsonl", "72B-generated")]
R = "/data/sakhaei/runs/"

def auc(pos, neg):
    if not pos or not neg: return None
    return sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg) / (len(pos)*len(neg))

def cell(rows, m):
    g = lambda s: [r["ratings"].get(m, {}).get("rating") for r in rows
                   if r["stratum"] == s and r["ratings"].get(m, {}).get("rating") is not None]
    dead, mid, gold = g("dead"), g("mid"), g("gold")
    if not (dead and mid and gold): return None
    return {"dead": st.mean(dead), "mid": st.mean(mid), "gold": st.mean(gold),
            "gap": st.mean(gold) - st.mean(mid),
            "auc_dead": auc(mid+gold, dead), "auc_gm": auc(gold, mid),
            "n": (len(dead), len(gold), len(mid))}

data = {}
for fn, gl in GENS:
    try:
        data[gl] = [json.loads(l) for l in open(R+fn)]
    except FileNotFoundError:
        data[gl] = None

def table(key, title, fmt="{:+.2f}"):
    print(f"\n{title}")
    hdr = f"  {'rater':24s}" + "".join(f"{gl:>16s}" for _, gl in GENS)
    print(hdr); print("  " + "-"*(len(hdr)-2))
    for m, label in RATERS:
        line = f"  {label:24s}"
        for _, gl in GENS:
            rows = data.get(gl)
            c = cell(rows, m) if rows else None
            line += f"{(fmt.format(c[key]) if c else 'n/a'):>16s}"
        print(line)

print("=" * 76)
print("SELF-RATED STEP INFORMATIVENESS -- rater (rows) x step generator (columns)")
for _, gl in GENS:
    rows = data.get(gl)
    if rows:
        ns = {s: sum(1 for r in rows if r['stratum'] == s) for s in ('dead','gold','mid')}
        print(f"  {gl}: n={len(rows)}  dead={ns['dead']} gold={ns['gold']} mid={ns['mid']}")
table("gap",      "GOLD minus MID margin (0-4 scale; negative = ranks the answer BELOW non-answers)")
table("auc_gm",   "AUC gold vs mid  -- 'was it actually the answer?'  (0.5 = coin flip)", "{:.3f}")
table("auc_dead", "AUC informative vs dead -- 'did anything come back?'", "{:.3f}")
table("mid",      "MID stratum mean -- how convincing this generator's WRONG steps look", "{:.2f}")
