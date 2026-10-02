#!/usr/bin/env python3
"""Per-step training metrics for Exp 1-4 from the trainer console logs (+ wandb offline for Exp 2 steps the log lost).
Dedup by step, keeping the LAST occurrence in chronological file order (resumes re-run steps). Output JSON."""
import re, glob, os, json, subprocess
T="/data/sakhaei/runs/train"
EXPS={
 "exp1": sorted(glob.glob(f"{T}/flowgrpo_qwen3_8b_judge[4-9]*/train.log"), key=os.path.getmtime),
 "exp2": sorted(glob.glob(f"{T}/flowgrpo_qwen3_8b_lr1e5_r*/train.log"), key=os.path.getmtime),
 "exp3": sorted(glob.glob(f"{T}/flowgrpo_qwen3_8b_lr3e6pen_*/train.log"), key=os.path.getmtime),
 "exp4": sorted(glob.glob(f"{T}/flowgrpo_qwen3_8b_rule_lr3e6pen_*/train.log"), key=os.path.getmtime),
}
KEYS={"reward":"critic/score/mean","kl":"actor/kl_loss","len":"response_length/mean","clip":"response_length/clip_ratio","drop":"n_dropped_sample_because_of_prompt","sec":"perf/time_per_step"}
out={}
for e,files in EXPS.items():
    rows={}
    for f in files:
        txt=subprocess.run(["grep","-a","critic/score/mean",f],capture_output=True,text=True).stdout
        for line in txt.splitlines():
            m=re.search(r"step:(\d+)",line)
            if not m: continue
            st=int(m.group(1)); r={"step":st,"src":os.path.basename(os.path.dirname(f))}
            for k,kk in KEYS.items():
                mm=re.search(re.escape(kk)+r":([0-9.e-]+)",line); r[k]=float(mm.group(1)) if mm else None
            rows[st]=r
    out[e]={"files":[os.path.basename(os.path.dirname(f)) for f in files],"rows":[rows[s] for s in sorted(rows)]}
    print(e, len(files),"files", len(rows),"steps", (min(rows),max(rows)) if rows else None)
# wandb offline: fill Exp 2 steps missing from the logs
try:
    from wandb.sdk.internal import datastore
    from wandb.proto import wandb_internal_pb2 as pb
    import yaml
    have={r["step"] for r in out["exp2"]["rows"]}
    for d in sorted(glob.glob("/data/sakhaei/runs/wandb/wandb/offline-run-*")+glob.glob("/data/sakhaei/runs/wandb/offline-run-*")):
        cfg=f"{d}/files/config.yaml"
        if not os.path.exists(cfg): continue
        c=yaml.safe_load(open(cfg)); name=json.dumps(c)
        if "flowgrpo_qwen3_8b_judge_lr1e5" not in name: continue
        ds=datastore.DataStore(); ds.open_for_scan(glob.glob(f"{d}/run-*.wandb")[0]); n=0
        while True:
            data=ds.scan_data()
            if data is None: break
            rec=pb.Record(); rec.ParseFromString(data)
            if rec.WhichOneof("record_type")!="history": continue
            h={it.key: json.loads(it.value_json) for it in rec.history.item}
            st=h.get("training/global_step") or h.get("_step")
            if st is None or "critic/score/mean" not in h: continue
            st=int(st)
            if st in have: continue
            out["exp2"]["rows"].append({"step":st,"src":"wandb:"+os.path.basename(d),**{k:h.get(kk) for k,kk in KEYS.items()}}); have.add(st); n+=1
        print("wandb", os.path.basename(d), "added", n)
    out["exp2"]["rows"].sort(key=lambda r:r["step"])
except Exception as ex:
    print("wandb parse skipped:", repr(ex)[:200])
json.dump(out, open("/data/sakhaei/runs/train_curves.json","w"))
for e in out: print(e, "steps:", [r["step"] for r in out[e]["rows"]][:3], "...", [r["step"] for r in out[e]["rows"]][-3:], "n=", len(out[e]["rows"]))
