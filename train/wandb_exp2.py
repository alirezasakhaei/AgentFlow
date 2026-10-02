import glob, sys, json, struct
sys.argv=[sys.argv[0],"x"]
exec(open("/tmp/wandb_offline.py").read().split("want=sys.argv[1]")[0])
from wandb.proto import wandb_internal_pb2 as pb
KEYS={"reward":"critic/score/mean","kl":"actor/kl_loss","len":"response_length/mean","clip":"response_length/clip_ratio","drop":"n_dropped_sample_because_of_prompt","sec":"perf/time_per_step"}
rows={}
for d in sorted(glob.glob("/data/sakhaei/runs/wandb/wandb/offline-run-202609*")):
    f=glob.glob(f"{d}/run-*.wandb")
    if not f: continue
    name=None; local={}
    for raw in records(f[0]):
        rec=pb.Record(); rec.ParseFromString(raw); t=rec.WhichOneof("record_type")
        if t=="run": name=rec.run.display_name
        if t=="history":
            h={(it.key or "/".join(it.nested_key)): it.value_json for it in rec.history.item}
            if "critic/score/mean" in h and h.get("training/global_step"):
                st=int(float(h["training/global_step"]))
                local[st]={"step":st,"src":"wandb:"+d.split("-")[-1],**{k:(float(h[kk]) if kk in h else None) for k,kk in KEYS.items()}}
    if name=="flowgrpo_qwen3_8b_judge_lr1e5" and local:
        print(d.split("/")[-1], "steps", min(local), max(local), len(local)); rows.update(local)   # later runs override earlier
json.dump(rows, open("/tmp/exp2_wandb_rows.json","w")); print("total", len(rows))
