#!/usr/bin/env python3
"""One clean wandb run per experiment, built from the local offline .wandb files (all verl metrics, every step; resumed
segments merged, the chronologically last segment wins per step). Upload + rename the live Exp 5 run. NO deletion here:
the raw segment/smoke runs are listed at the end for Alireza to delete."""
import glob, os, json, struct, time
from wandb.proto import wandb_internal_pb2 as pb
import wandb
BLOCK=32768
def records(path):
    data=open(path,"rb").read(); pos=7; buf=b""
    while pos+7<=len(data):
        if BLOCK-(pos%BLOCK)<7: pos+=BLOCK-(pos%BLOCK); continue
        crc,ln,typ=struct.unpack("<IHB",data[pos:pos+7]); pos+=7; chunk=data[pos:pos+ln]; pos+=ln
        if typ==1: yield chunk
        elif typ==2: buf=chunk
        elif typ==3: buf+=chunk
        elif typ==4: buf+=chunk; yield buf; buf=b""
        else: break
ENTITY="alirezasakhaeirad"; PROJECT="MAReasoning"
EXPS={
 "flowgrpo_qwen3_8b_judge":           ("exp1","Exp 1 — Flow-GRPO replication, Qwen3-8B, lr 1e-6, judge reward",{"lr":1e-6,"reward":"Qwen3-30B judge","planner":"Qwen3-8B","penalty":False,"max_prompt":6144}),
 "flowgrpo_qwen3_8b_judge_lr1e5":     ("exp2","Exp 2 — Qwen3-8B, lr 1e-5, judge reward (collapsed at ~47)",{"lr":1e-5,"reward":"Qwen3-30B judge","planner":"Qwen3-8B","penalty":False,"max_prompt":6144}),
 "flowgrpo_qwen3_8b_judge_lr3e6_pen": ("exp3","Exp 3 — Qwen3-8B, lr 3e-6, overlong penalty, judge reward",{"lr":3e-6,"reward":"Qwen3-30B judge","planner":"Qwen3-8B","penalty":True,"max_prompt":8192}),
 "flowgrpo_qwen3_8b_rule_lr3e6_pen":  ("exp4","Exp 4 — Qwen3-8B, lr 3e-6, overlong penalty, rule reward",{"lr":3e-6,"reward":"rule scorer","planner":"Qwen3-8B","penalty":True,"max_prompt":8192}),
}
merged={k:{} for k in EXPS}
for d in sorted(glob.glob("/data/sakhaei/runs/wandb/wandb/offline-run-*")):
    f=glob.glob(f"{d}/run-*.wandb")
    if not f: continue
    name=None; hist=[]
    for raw in records(f[0]):
        rec=pb.Record(); rec.ParseFromString(raw); t=rec.WhichOneof("record_type")
        if t=="run": name=rec.run.display_name
        if t=="history":
            h={(it.key or "/".join(it.nested_key)): json.loads(it.value_json) for it in rec.history.item}
            if "critic/score/mean" in h and "training/global_step" in h: hist.append(h)
    if name in EXPS:
        for h in hist: merged[name][int(h["training/global_step"])]=h
        print(os.path.basename(d), name, "steps", len(hist))
keep={"82aiyxat"}
for name,(rid,clean,cfg) in EXPS.items():
    rows=merged[name]
    if not rows: print("no data for", name); continue
    run=wandb.init(entity=ENTITY, project=PROJECT, id=rid, name=clean, resume="allow", config=cfg, reinit=True)
    for st in sorted(rows):
        run.log({k:v for k,v in rows[st].items() if not k.startswith("_") and isinstance(v,(int,float))}, step=st)
    run.summary["steps_logged"]=len(rows); run.finish(); keep.add(rid)
    print("uploaded", rid, clean, "steps", min(rows), max(rows), len(rows))
time.sleep(5); api=wandb.Api()
live=api.run(f"{ENTITY}/{PROJECT}/82aiyxat"); live.name="Exp 5 — upstream AgentFlow as published, Qwen2.5-7B-Instruct"; live.update(); print("renamed live run")
print("KEEP:", sorted(keep)); print("TO DELETE (raw segments / smokes):")
for r in api.runs(f"{ENTITY}/{PROJECT}"):
    if r.id not in keep: print("  ", r.id, r.name)
