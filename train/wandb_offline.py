#!/usr/bin/env python3
"""Read wandb offline .wandb files (LevelDB-style log of protobuf Records) without the removed datastore module."""
import struct, glob, os, sys, json, zlib
from wandb.proto import wandb_internal_pb2 as pb
BLOCK=32768
def records(path):
    data=open(path,"rb").read(); pos=7  # 7-byte datastore header
    buf=b""
    while pos+7<=len(data):
        if BLOCK-(pos%BLOCK)<7: pos+=BLOCK-(pos%BLOCK); continue
        crc,ln,typ=struct.unpack("<IHB",data[pos:pos+7]); pos+=7
        chunk=data[pos:pos+ln]; pos+=ln
        if typ==1: yield chunk
        elif typ==2: buf=chunk
        elif typ==3: buf+=chunk
        elif typ==4: buf+=chunk; yield buf; buf=b""
        else: break
KEYS={"reward":"critic/score/mean","kl":"actor/kl_loss","len":"response_length/mean","clip":"response_length/clip_ratio","drop":"n_dropped_sample_because_of_prompt","sec":"perf/time_per_step"}
want=sys.argv[1]  # experiment-name substring
out={}
for d in sorted(glob.glob("/data/sakhaei/runs/wandb/wandb/offline-run-*")):
    f=glob.glob(f"{d}/run-*.wandb")
    if not f: continue
    name=None; rows={}
    try:
        for raw in records(f[0]):
            rec=pb.Record(); rec.ParseFromString(raw); t=rec.WhichOneof("record_type")
            if t=="run": name=rec.run.display_name or rec.run.run_id
            if t=="config":
                for it in rec.config.update:
                    if "experiment_name" in it.key or want in it.value_json: name=(name or "")+" | "+it.value_json[:80]
            if t=="history":
                h={it.key: json.loads(it.value_json) for it in rec.history.item}
                if "critic/score/mean" in h:
                    st=int(h.get("training/global_step", h.get("_step",0)))
                    rows[st]={"step":st,**{k:h.get(kk) for k,kk in KEYS.items()}}
    except Exception as e:
        print(os.path.basename(d),"ERR",repr(e)[:80]); continue
    cfg=open(f"{d}/files/config.yaml").read() if os.path.exists(f"{d}/files/config.yaml") else ""
    hit = want in cfg or (name and want in name)
    print(os.path.basename(d), "name=",(name or "")[:60], "steps=",(min(rows),max(rows),len(rows)) if rows else None, "HIT" if hit else "")
    if hit and rows: out[os.path.basename(d)]=rows
json.dump(out,open("/tmp/wandb_hits.json","w"))
print("saved", {k:len(v) for k,v in out.items()})
