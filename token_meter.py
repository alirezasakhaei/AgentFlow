#!/usr/bin/env python3
"""Snapshot / diff vLLM token counters across every pod in the topology.

The harness rule is that a result without helper-pod tokens is not a result, and that arms
are compared by token budget rather than turn count. vLLM exposes cumulative
prompt/generation counters on /metrics, so a snapshot before and after a run gives exact
totals for BOTH the planner under test (:8000) and the shared helper pod (:8001).

Caveat: the counters are per-server and cumulative, so a diff is only attributable to this
run if nothing else is hitting the same ports meanwhile. Check nvidia-smi / who first.

  python token_meter.py snap  before.json
  python token_meter.py diff  before.json after.json
"""
import json, sys, urllib.request

PODS = {
    "planner  (mds1:8000)": "http://localhost:8000/metrics",
    "helpers  (mds1:8001)": "http://localhost:8001/metrics",
    # PATCHED (MAReasoning): the 32B/72B planner arms serve from mds2's H100, so a
    # mds1-only ledger reported their planner cost as zero. Unreachable pods are recorded
    # as an error and simply contribute nothing, so this is safe when mds2 is down.
    "planner  (mds2:8000)": "http://mds1srv2.epfl.ch:8000/metrics",
}
KEYS = ("vllm:prompt_tokens_total", "vllm:generation_tokens_total",
        "vllm:request_success_total")

def scrape(url):
    out = {}
    try:
        raw = urllib.request.urlopen(url, timeout=15).read().decode()
    except Exception as e:
        return {"error": str(e)}
    for line in raw.splitlines():
        if line.startswith("#"):
            continue
        for k in KEYS:
            if line.startswith(k):
                name, _, val = line.rpartition(" ")
                try:
                    out[name] = out.get(name, 0.0) + float(val)
                except ValueError:
                    pass
    agg = {}
    for name, v in out.items():
        base = name.split("{")[0]
        agg[base] = agg.get(base, 0.0) + v
    return agg

def snap():
    return {pod: scrape(url) for pod, url in PODS.items()}

def main():
    mode = sys.argv[1]
    if mode == "snap":
        json.dump(snap(), open(sys.argv[2], "w"), indent=2)
        print(f"snapshot -> {sys.argv[2]}")
    elif mode == "diff":
        a = json.load(open(sys.argv[2]))
        b = json.load(open(sys.argv[3]))
        grand = 0.0
        print(f"\n=== token usage ({sys.argv[2]} -> {sys.argv[3]}) ===")
        for pod in PODS:
            pa, pb = a.get(pod, {}), b.get(pod, {})
            p = pb.get("vllm:prompt_tokens_total", 0) - pa.get("vllm:prompt_tokens_total", 0)
            g = pb.get("vllm:generation_tokens_total", 0) - pa.get("vllm:generation_tokens_total", 0)
            r = pb.get("vllm:request_success_total", 0) - pa.get("vllm:request_success_total", 0)
            grand += p + g
            print(f"{pod:16s} prompt {p:12,.0f}  generation {g:10,.0f}  total {p+g:12,.0f}  requests {r:7,.0f}")
        print(f"{'ALL PODS':16s} {'':31s} total {grand:12,.0f}   <- the number to compare arms by")
    else:
        sys.exit("usage: token_meter.py snap OUT.json | diff BEFORE.json AFTER.json")

if __name__ == "__main__":
    main()
