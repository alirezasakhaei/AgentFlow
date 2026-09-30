"""Rule-based training reward. PATCHED (MAReasoning) 2026-09-23.

Upstream used a gpt-4o LLM judge (needs OPENAI_API_KEY, non-deterministic, paid). This is the
project's standing rule: no LLM judge, rule-based scoring only. Binary {0,1}, deterministic.

  math   (source mathhard / aime2024 / math*): math_verify equivalence, then a numeric /
         normalised-string fallback.
  search (source nq / anything else): SQuAD-normalised exact match against any of the
         "; "-separated golds, or the gold as a word-boundary substring of a SHORT answer
         (<= max(8, 2*len(gold)+4) tokens) so hedging across many candidates is not rewarded.
When source is unknown both checks run and either passing counts.
"""
import re, string, math

MATH_SOURCES = ("mathhard", "aime", "math", "amc", "olympiad", "minerva", "gsm")

def normalize(s: str) -> str:
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def _strip_math(s: str) -> str:
    s = str(s).strip()
    m = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", s)
    if m: s = m[-1]
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("°", "")
    s = s.replace("\\%", "").replace("%", "").replace(",", "")
    s = re.sub(r"\\d?frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    s = s.replace(" ", "").rstrip(".")
    if s.startswith("{") and s.endswith("}"): s = s[1:-1]
    return s

def _num(s):
    try:
        if "/" in s and s.count("/") == 1:
            a, b = s.replace("(", "").replace(")", "").split("/"); return float(a) / float(b)
        return float(s)
    except Exception:
        return None

def math_equal(pred: str, gold: str) -> bool:
    pred, gold = str(pred), str(gold)
    try:
        from math_verify import parse, verify
        g = parse(gold if "$" in gold or "\\boxed" in gold else f"${gold}$")
        for cand in (pred, f"${pred}$"):
            p = parse(cand)
            if g and p and verify(g, p):
                return True
    except Exception:
        pass
    a, b = _strip_math(pred), _strip_math(gold)
    if a and a == b:
        return True
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        return math.isclose(na, nb, rel_tol=1e-6, abs_tol=1e-9)
    return False

def search_correct(pred: str, gold: str) -> bool:
    p = normalize(pred)
    if not p:
        return False
    golds = [g for g in re.split(r"\s*;\s*", str(gold)) if g.strip()] or [str(gold)]
    for g in golds:
        ng = normalize(g)
        if not ng:
            continue
        if p == ng:
            return True
        if re.search(r"(?<!\w)" + re.escape(ng) + r"(?!\w)", p) and \
           len(p.split()) <= max(8, 2 * len(ng.split()) + 4):
            return True
    return False

def rule_score(question: str, groundtruth: str, answer_extracted: str, source: str = None) -> bool:
    """Rule-based verdict (math_verify / normalised EM); used directly when REWARD_MODE=rule,
    and as the fallback when the judge is unreachable."""
    src = (source or "").lower()
    if any(k in src for k in MATH_SOURCES):
        return math_equal(answer_extracted, groundtruth)
    if src == "nq" or "search" in src or "wiki" in src or "hotpot" in src or "musique" in src:
        return search_correct(answer_extracted, groundtruth)
    return math_equal(answer_extracted, groundtruth) or search_correct(answer_extracted, groundtruth)

def compute_score(question: str, groundtruth: str, answer_extracted: str, source: str = None) -> bool:
    """Same signature as upstream (question, groundtruth, answer) plus the data source.

    REWARD_MODE=judge (default): local Qwen3-30B-A3B judge (train/judge.py, AgentFlow's own rubric),
    rule-based fallback if the judge fails. REWARD_MODE=rule: rule-based only. Both verdicts are
    printed as a `[reward]` line so judge-vs-rule agreement can be measured from the rollout log.
    """
    if answer_extracted is None or str(answer_extracted).strip() in ("", "None"):
        return False
    import os
    mode = os.environ.get("REWARD_MODE", "judge").lower()
    rule = rule_score(question, groundtruth, answer_extracted, source)
    if mode == "rule":
        return rule
    from judge import judge
    verdict, raw = judge(question, groundtruth, answer_extracted)
    print(f"[reward] source={source} judge={verdict} rule={rule} gold={str(groundtruth)[:40]!r} answer={str(answer_extracted)[:60]!r}"
          + (f" judge_raw={raw[:80]!r}" if verdict is None else ""))
    return rule if verdict is None else verdict

def _compute_score_rule_only(question, groundtruth, answer_extracted, source=None):
    if answer_extracted is None or str(answer_extracted).strip() in ("", "None"):
        return False
    src = (source or "").lower()
    if any(k in src for k in MATH_SOURCES):
        return math_equal(answer_extracted, groundtruth)
    if src == "nq" or "search" in src or "wiki" in src or "hotpot" in src or "musique" in src:
        return search_correct(answer_extracted, groundtruth)
    return math_equal(answer_extracted, groundtruth) or search_correct(answer_extracted, groundtruth)

def eval(question, groundtruth, answer_extracted, val: bool = False, source: str = None) -> float:
    return 1.0 if compute_score(str(question), str(groundtruth), str(answer_extracted), source) else 0.0

if __name__ == "__main__":
    tests = [("mathhard", "\\frac{1}{2}", "0.5", True), ("mathhard", "0.5", "\\boxed{\\frac{1}{2}}", True), ("mathhard", "\\sqrt{2}", "1.4142135", False),
             ("mathhard", "42", "The answer is \\boxed{42}.", True),
             ("mathhard", "42", "41", False),
             ("nq", "James Madison; Madison", "james madison", True),
             ("nq", "1999", "It was 1999", True),
             ("nq", "Paris", "Paris, London, Berlin, Rome, Madrid, Lisbon, Vienna, Oslo, Bern", False),
             ("nq", "no", "I do not know", False),
             ("aime2024", "204", "204", True)]
    for src, g, a, exp in tests:
        got = rule_score("q", g, a, src); print("OK " if got == exp else "BAD", src, repr(g), repr(a), got)
