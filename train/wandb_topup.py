"""Append the few steps the wandb records lack (trainer crashed before flushing) from the console-log extraction, 5 core metrics."""
import json, wandb
ENTITY="alirezasakhaeirad"; PROJECT="MAReasoning"
cur=json.load(open("/tmp/train_curves.json"))
KEYS={"reward":"critic/score/mean","kl":"actor/kl_loss","len":"response_length/mean","clip":"response_length/clip_ratio","drop":"n_dropped_sample_because_of_prompt","sec":"perf/time_per_step"}
api=wandb.Api()
for e in ["exp1","exp2","exp3","exp4"]:
    r=api.run(f"{ENTITY}/{PROJECT}/{e}"); have={int(h["_step"]) for h in r.scan_history(keys=["_step","critic/score/mean"])}
    rows=[x for x in cur[e]["rows"] if x["step"] not in have and x["reward"] is not None]
    if not rows: print(e,"complete"); continue
    run=wandb.init(entity=ENTITY, project=PROJECT, id=e, resume="must", reinit=True)
    for x in sorted(rows,key=lambda x:x["step"]): run.log({kk:x[k] for k,kk in KEYS.items() if x.get(k) is not None}, step=x["step"])
    run.finish(); print(e,"appended steps",[x["step"] for x in rows])
