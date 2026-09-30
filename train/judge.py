"""LLM-judge training reward on a local Qwen3-30B-A3B (MAReasoning, 2026-09-23).

Restores AgentFlow's judge-based reward (train/utils.py upstream used gpt-4o with the same
rubric) but on a self-hosted model, so the recipe matches the paper without a paid key.
The rule-based check in train/utils.py stays as the fallback when the judge is unreachable or
returns garbage, and its verdict is logged next to the judge's so the two can be compared.
Eval scoring elsewhere stays rule-based; this is the *training* reward only.
"""
import os, re, json, urllib.request

JUDGE_URL = os.environ.get("JUDGE_URL", "http://mds1srv2.epfl.ch:8001/v1/chat/completions")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-30B-A3B-judge")
JUDGE_KEY = os.environ.get("VLLM_API_KEY", "dummy-token")

PROMPT = """You are a precise evaluator. Determine if the Model Response is equivalent to the Ground Truth.

**Instructions:**
1.  **Extract:** Isolate the final answer from the Model Response, ignoring reasoning. Look for `\\boxed{{...}}` or concluding statements.
2.  **Normalize & Compare:** The extracted answer and Ground Truth must be equivalent after normalization:
    - **Math:** Mathematically identical (e.g., `\\frac{{1}}{{2}}` == `0.5`).
    - **Numbers/Text:** Ignore formatting, case, and currency/units (e.g., `1,000` == `1000`).
    - **MCQ:** Match option content (e.g., "Paris") or number (e.g., `3rd` option) to the correct letter.
3.  **Verdict:** "True" only for semantically or mathematically equivalent answers. A response that hedges across several different candidate answers is NOT equivalent.

**Inputs:**
Question: {question}
Model Response: {answer}
Ground Truth: {gold}

Reply with exactly two lines:
analysis: <one sentence>
verdict: True or False"""

def judge(question: str, gold: str, answer: str, timeout: float = 120.0):
    """Returns (verdict: bool|None, raw_text). None means the judge could not be used."""
    body = {"model": JUDGE_MODEL, "temperature": 0.0, "max_tokens": 120,
            "messages": [{"role": "user", "content": PROMPT.format(question=str(question)[:3000], answer=str(answer)[:2000], gold=str(gold)[:500])}]}
    try:
        req = urllib.request.Request(JUDGE_URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {JUDGE_KEY}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt = json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:
        return None, f"ERR {type(e).__name__}: {e}"
    m = re.search(r"verdict\s*:\s*\**\s*(true|false)", txt, re.I)
    if not m:
        m = re.search(r"\b(true|false)\b", txt.strip().splitlines()[-1] if txt.strip() else "", re.I)
    return (m.group(1).lower() == "true") if m else None, txt

if __name__ == "__main__":
    for q, g, a in [("What is 1/2 as a decimal?", "\\frac{1}{2}", "0.5"),
                    ("Who was president when Citibank was founded?", "james madison", "James Madison"),
                    ("Capital?", "Paris", "Paris, London, Berlin, Rome, Madrid, Lisbon, Vienna, Oslo, Bern"),
                    ("Is it true?", "no", "I do not know"),
                    ("Value?", "42", "\\boxed{41}")]:
        v, raw = judge(q, g, a); print(v, "|", raw.replace("\n", " / ")[:140])
