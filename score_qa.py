#!/usr/bin/env python3
"""Rule-based scorer for the short-answer QA tasks (2wiki, bamboogle, hotpotqa, musique).

No LLM judge.

The shipped calculate_score_unified.py instantiates a gpt-4o judge and refuses to run
without OPENAI_API_KEY; bamboogle ships short reference strings ("james madison",
"Titan IIIE"), so SQuAD-style normalised exact match plus token F1 is enough and is
reproducible. Mirrors score_gameof24.py in spirit: the primary number is strict, and a
looser "cover" number is reported separately as a ceiling, never as the score.
"""
import argparse, json, os, re, string, collections

def normalize(s):
    """SQuAD normalisation: lowercase, drop punctuation, articles and extra whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def extract_prediction(text):
    """Pull the model's final answer out of direct_output.

    The bamboogle prompts explicitly ask for <answer>...</answer>, so that tag is
    authoritative when present; the fallbacks only cover models that ignored the format.
    """
    if not text:
        return ""
    m = re.findall(r"<answer>(.*?)</answer>", text, re.S | re.I)
    block = m[-1].strip() if m else text.strip()
    # AgentFlow's final-output prompt makes the planner emit a whole "Process Summary"
    # inside the answer span, ending in an explicit "Answer:" line. Strict EM against a
    # short gold ("james madison") fails on the prose, so take the text after the last
    # Answer:/Final answer: marker when one is present. Marker-driven, not judged.
    mk = re.findall(r"(?:final answer|answer)\s*\**\s*[::]\s*\**\s*(.+)", block, re.I)
    if mk:
        return mk[-1].strip().split("\n")[0].strip(" *`\"")
    if m:
        return block.split("\n")[-1].strip(" *`\"") if "\n" in block else block
    return block.split("\n")[-1].strip(" *`\"")

def contains_all_gold_tokens(pred, gold):
    """Relaxed match: every normalised gold token appears in the prediction.

    Bridges the two failure modes of the strict metrics on this harness. EM is ~0 because the
    planner answers in prose; plain substring `cover` misses "David N. Dinkins" vs gold
    "David Dinkins". Containment catches both, stays rule-based and reproducible, but
    over-credits a prediction that hedges across several candidate answers -- so it is
    reported alongside EM, never instead of it.
    """
    p_tokens, g_tokens = set(normalize(pred).split()), normalize(gold).split()
    return int(bool(g_tokens) and all(t in p_tokens for t in g_tokens))

def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision, recall = n / len(p), n / len(g)
    return 2 * precision * recall / (precision + recall)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--verbose", action="store_true", help="print every item")
    args = ap.parse_args()

    data = json.load(open(args.data_file))
    golds = {str(d.get("pid", d.get("idx"))): d["answer"] for d in data}

    n = em_n = 0
    f1_sum = 0.0
    cover_n = contain_n = 0
    steps, times = [], []
    tool_counter = collections.Counter()
    step1_tool = collections.Counter()
    missing, empty = [], []

    for pid in sorted(golds, key=lambda x: int(x)):
        path = os.path.join(args.result_dir, f"output_{pid}.json")
        if not os.path.exists(path):
            missing.append(pid)
            continue
        try:
            d = json.load(open(path))
        except Exception:
            missing.append(pid)
            continue
        n += 1
        gold_list = golds[pid] if isinstance(golds[pid], list) else [golds[pid]]
        pred = extract_prediction(d.get("direct_output", ""))
        if not pred:
            empty.append(pid)
        em = max(int(normalize(pred) == normalize(g)) for g in gold_list)
        f1 = max(token_f1(pred, g) for g in gold_list)
        # "cover" = the gold string appears anywhere in the final output. A ceiling only:
        # it credits answers the model never actually committed to.
        blob = normalize(d.get("direct_output", ""))
        # PATCHED (MAReasoning) 2026-09-18: word-boundary match. Plain substring let gold
        # "no" hit inside "not"/"know", inflating cover on yes/no items.
        cover = max(int(bool(re.search(r"(?<!\w)" + re.escape(normalize(g)) + r"(?!\w)", blob)))
                    for g in gold_list)
        contain = max(contains_all_gold_tokens(pred, g) for g in gold_list)
        em_n += em
        f1_sum += f1
        cover_n += cover
        contain_n += contain
        steps.append(d.get("step_count", 0))
        times.append(d.get("execution_time", 0.0))
        mem = d.get("memory", {}) or {}
        tools_used = [v.get("tool_name", "?") for v in mem.values() if isinstance(v, dict)]
        for t in tools_used:
            tool_counter[t] += 1
        if tools_used:
            step1_tool[tools_used[0]] += 1
        if args.verbose:
            print(f"[{pid}] em={em} f1={f1:.2f} cover={cover} contain={contain} pred={pred[:60]!r} gold={gold_list}")

    print(f"\n=== {os.path.basename(os.path.dirname(os.path.dirname(args.result_dir.rstrip(chr(47)))))} | {args.result_dir} ===")
    print(f"scored items        : {n} / {len(golds)}")
    if missing:
        print(f"missing outputs     : {len(missing)} -> {missing[:12]}{' ...' if len(missing) > 12 else ''}")
    if empty:
        print(f"empty predictions   : {len(empty)} -> {empty[:12]}{' ...' if len(empty) > 12 else ''}")
    if n:
        print(f"exact match         : {em_n}/{n} = {100.0 * em_n / n:.1f}%")
        print(f"token F1            : {100.0 * f1_sum / n:.1f}%")
        print(f"gold-token contained: {contain_n}/{n} = {100.0 * contain_n / n:.1f}%   <- relaxed, over-credits hedging")
        print(f"cover (ceiling)     : {cover_n}/{n} = {100.0 * cover_n / n:.1f}%   <- not a score")
        print(f"mean steps          : {sum(steps) / n:.2f}")
        print(f"mean s/item         : {sum(times) / n:.1f}")
        print("\ntool calls (all steps):")
        for t, c in tool_counter.most_common():
            print(f"   {c:5d}  {t}")
        print("first tool chosen per item:")
        for t, c in step1_tool.most_common():
            print(f"   {c:5d}  {t}")
        wiki = sum(c for t, c in tool_counter.items() if "Wikipedia" in t)
        print(f"\nitems whose FIRST tool was Wikipedia: {step1_tool.get('Wikipedia_RAG_Search_Tool', 0)}/{n}")
        print(f"total Wikipedia tool calls          : {wiki}")

if __name__ == "__main__":
    main()
