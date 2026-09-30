"""Deterministic scorer for gameof24 — no LLM judge.

Correct iff a paren-balanced arithmetic expression appears in the model's FINAL
ANSWER SECTION that (a) uses exactly the multiset of the four given numbers and
(b) evaluates to exactly 24 over the rationals.

`any_valid_upper_bound` accepts such an expression anywhere in the output,
including ones the model itself rejected — an upper bound, not a score.
"""
import argparse, ast, json, os, re
from fractions import Fraction

ANSWER_MARKERS = re.compile(
    r"(?:final\s+answer|answer|solution)\s*(?:is)?\s*[:\*]*|\\boxed", re.IGNORECASE)


def de_latex(text: str) -> str:
    t = text
    for a, b in (("\u00d7", "*"), ("\u00f7", "/"), ("\u2212", "-"), ("\u00b7", "*")):
        t = t.replace(a, b)
    t = re.sub(r"\\(?:times|cdot|ast)\b", "*", t)
    t = re.sub(r"\\(?:div|over)\b", "/", t)
    # \frac{a}{b} -> ((a)/(b)), innermost-first so nesting resolves
    for _ in range(6):
        new = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", t)
        if new == t:
            break
        t = new
    t = re.sub(r"\\boxed\s*\{", " ( ", t)
    t = re.sub(r"\\(?:left|right|displaystyle|mathbf|text|colorbox|quad|;|,|!)\b", " ", t)
    t = re.sub(r"\\[\[\]()]", " ", t)
    t = t.replace("$", " ").replace("{", " ").replace("}", " ")
    t = re.sub(r"\\[a-zA-Z]+", " ", t)          # any remaining latex command
    return t


def safe_eval(expr):
    try:
        node = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, bool) or not isinstance(n.value, (int, float)):
                raise ValueError
            return Fraction(str(n.value))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = ev(n.operand)
            return v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.BinOp):
            l, r = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return l + r
            if isinstance(n.op, ast.Sub):
                return l - r
            if isinstance(n.op, ast.Mult):
                return l * r
            if isinstance(n.op, ast.Div):
                if r == 0:
                    raise ZeroDivisionError
                return l / r
        raise ValueError

    try:
        return ev(node)
    except Exception:
        return None


ALLOWED = re.compile(r"[0-9\+\-\*/\(\)\. ]+")


def balanced_subexprs(run: str):
    """Yield paren-balanced candidate expressions from a run of arithmetic chars."""
    run = run.strip()
    if not run:
        return
    starts = [i for i, c in enumerate(run) if c.isdigit() or c == "("]
    ends = [i for i, c in enumerate(run) if c.isdigit() or c == ")"]
    seen = set()
    for i in starts[:40]:
        for j in reversed(ends[-40:]):
            if j < i + 2:
                continue
            sub = run[i:j + 1].replace(" ", "")
            if sub in seen or not any(op in sub for op in "+-*/"):
                continue
            if sub.count("(") != sub.count(")"):
                continue
            seen.add(sub)
            yield sub


def numbers_in(expr):
    return sorted(int(x) for x in re.findall(r"\d+", expr))


def valid_expressions(text, target):
    """All expressions in `text` using exactly `target`'s numbers and equal to 24."""
    out = []
    for run in ALLOWED.findall(de_latex(text)):
        for sub in balanced_subexprs(run):
            if numbers_in(sub) != sorted(target):
                continue
            if safe_eval(sub) == 24:
                out.append(sub)
    return out


def answer_section(text):
    """Text after the last answer marker; falls back to the last 25% of the output."""
    marks = list(ANSWER_MARKERS.finditer(text))
    if marks:
        return text[marks[-1].start():]
    return text[int(len(text) * 0.75):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--output_file", default="score_rulebased.json")
    a = ap.parse_args()

    data = {str(d["pid"]): d for d in json.load(open(a.data_file))}
    rows, ok_n, any_n = [], 0, 0
    files = sorted((f for f in os.listdir(a.result_dir) if re.fullmatch(r"output_\d+\.json", f)),
                   key=lambda f: int(re.findall(r"\d+", f)[0]))
    for f in files:
        d = json.load(open(os.path.join(a.result_dir, f)))
        nums = data[str(d["pid"])]["question"]
        text = str(d.get("direct_output", ""))
        in_answer = valid_expressions(answer_section(text), nums)
        anywhere = in_answer or valid_expressions(text, nums)
        ok_n += bool(in_answer)
        any_n += bool(anywhere)
        rows.append({"pid": d["pid"], "numbers": nums,
                     "correct": bool(in_answer), "any_valid": bool(anywhere),
                     "expr": in_answer[0] if in_answer else (anywhere[0] if anywhere else None),
                     "steps": d.get("step_count"), "time_s": d.get("execution_time"),
                     "out_chars": len(text)})
    n = len(rows) or 1
    summary = {"n": len(rows),
               "accuracy": round(ok_n / n, 4),
               "any_valid_upper_bound": round(any_n / n, 4),
               "mean_steps": round(sum(r["steps"] or 0 for r in rows) / n, 2),
               "mean_time_s": round(sum(r["time_s"] or 0 for r in rows) / n, 1)}
    json.dump({"summary": summary, "rows": rows},
              open(os.path.join(a.result_dir, a.output_file), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
