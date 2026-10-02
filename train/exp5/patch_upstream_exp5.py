#!/usr/bin/env python3
"""Exp 5 (2026-10-02): the minimal set of edits to the pristine upstream AgentFlow clone that a NO-PAID-KEY run needs.
Every edit is marked 'PATCHED (MAReasoning, Exp5)'. Nothing else in the clone is touched; hyper-parameters live in train/config.yaml."""
import re, shutil, os
U="/data/sakhaei/code/AgentFlow-upstream"; F="/data/sakhaei/code/AgentFlow"
def edit(path, old, new, count=1):
    s=open(path).read(); assert old in s, (path, old[:60]); s=s.replace(old,new,count); open(path,"w").write(s); print("patched", path)

# 1) factory.py: honour VLLM_BASE_URL so the frozen helpers can live on mds2 (upstream hard-codes localhost:8000)
edit(f"{U}/agentflow/agentflow/engine/factory.py",
 '"base_url": kwargs.get("base_url", "http://localhost:8000/v1"),',
 '"base_url": kwargs.get("base_url", os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")),  # PATCHED (MAReasoning, Exp5): helpers served on another host; upstream default kept when the env var is unset')
s=open(f"{U}/agentflow/agentflow/engine/factory.py").read()
if not re.search(r"^import os", s, re.M): open(f"{U}/agentflow/agentflow/engine/factory.py","w").write("import os  # PATCHED (MAReasoning, Exp5)\n"+s)

# 2) train/utils.py: the gpt-4o judge -> the same rubric on the locally served Qwen2.5-7B-Instruct (no OpenAI key). Same prompt text.
p=f"{U}/train/utils.py"; s=open(p).read()
s=s.replace('''try:
    llm_scorer_engine = ChatOpenAI(
        model_string="gpt-4o", 
        is_multimodal=False, 
        enable_cache=True
    )
    print(f"\\nLLM Scorer engine '{llm_scorer_engine.model_string}' initialized successfully.\\n")
except Exception as e:
    print(f"Failed to initialize LLM Scorer engine: {e}")
    llm_scorer_engine = None''',
'''# PATCHED (MAReasoning, Exp5): no OpenAI key -> the judge is the locally served Qwen2.5-7B-Instruct (the paper's own
# backbone) through the vLLM engine, with the UNCHANGED rubric below. The vLLM engine ignores response_format, so the
# verdict is parsed from the text (<true_false>: True/False) instead of a pydantic object.
import os
from agentflow.engine.vllm import ChatVLLM
try:
    llm_scorer_engine = ChatVLLM(model_string=os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
                                 base_url=os.environ.get("JUDGE_URL", os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")),
                                 is_multimodal=False, enable_cache=False, temperature=0.0, check_model=False)
    print(f"\\nLLM Scorer engine '{llm_scorer_engine.model_string}' (local vLLM) initialized successfully.\\n")
except Exception as e:
    print(f"Failed to initialize LLM Scorer engine: {e}")
    llm_scorer_engine = None''')
s=s.replace('''    verification_result = llm_scorer_engine(query_prompt, response_format=AnswerVerification)
    
    return verification_result.true_false''',
'''    verification_result = llm_scorer_engine(query_prompt, response_format=AnswerVerification)
    # PATCHED (MAReasoning, Exp5): parse the text verdict (vLLM engine returns a string)
    if isinstance(verification_result, AnswerVerification):
        return verification_result.true_false
    text = str(verification_result)
    m = re.search(r"true_false[^A-Za-z]*(True|False)", text, re.I)
    if m:
        return m.group(1).lower() == "true"
    m = re.findall(r"\\b(True|False)\\b", text)
    return bool(m) and m[-1].lower() == "true"''')
open(p,"w").write(s); print("patched", p)

# 3) runner.py: keep only trainable-planner triplets (helper calls through vLLM also become spans, without token ids)
edit(f"{U}/agentflow/runner.py",
 "            triplets = self.triplet_exporter.export(trace_spans)\n",
 '''            triplets = self.triplet_exporter.export(trace_spans)
            # PATCHED (MAReasoning, Exp5): the frozen helpers are served by a plain vLLM through the same openai
            # client, so their calls also become openai.chat.completion spans, but without token ids. Upstream never
            # saw this (DashScope/gpt-4o-mini helpers). Keep only trainable-planner calls and carry the terminal
            # reward over to the last kept triplet; otherwise final_reward is None and the daemon fills it with 0.
            if triplets:
                seen_rewards = [t.reward for t in triplets if t.reward is not None]
                kept = [t for t in triplets if t.prompt.get("token_ids") and t.response.get("token_ids")]
                if seen_rewards and kept and kept[-1].reward is None:
                    kept[-1] = kept[-1].model_copy(update={"reward": seen_rewards[-1]})
                if len(kept) != len(triplets):
                    logger.info(f"[Triplets] kept {len(kept)}/{len(triplets)} LLM calls with token ids (helper calls dropped)")
                triplets = kept
''')

# 4) Wikipedia + web-search tools: upstream needs OpenAI embeddings + gpt-4o-mini and hard-exits without OPENAI_API_KEY.
#    Take our fork's versions (local embedder, Wikimedia user-agent/robot fixes, lazy import). Behaviour, not recipe.
for rel in ["agentflow/agentflow/tools/wikipedia_search/tool.py","agentflow/agentflow/tools/web_search/tool.py"]:
    shutil.copy(f"{F}/{rel}", f"{U}/{rel}"); print("copied fork tool ->", rel)
print("done")
