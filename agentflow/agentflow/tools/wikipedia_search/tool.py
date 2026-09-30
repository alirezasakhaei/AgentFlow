import os
import sys
import wikipedia
from pydantic import BaseModel

from agentflow.tools.base import BaseTool
from agentflow.engine.factory import create_llm_engine
# PATCHED (MAReasoning): module-scope import removed. Initializer._load_single_tool
# registers every *Tool class in a module, so importing Web_Search_Tool here injected
# Web_RAG_Search_Tool (paid OpenAI embeddings) into the toolbox even when only
# Wikipedia_Search_Tool was enabled. Imported inside execute() instead.

# from web_rag import Web_Search_Tool
# from agentflow.tools.web_search.tool import Web_Search_Tool # NOTE: Shall be used in the future

# from utilis import select_relevant_queries

from agentflow.tools.base import BaseTool
from agentflow.engine.factory import create_llm_engine

# PATCHED (MAReasoning): Wikimedia now 403s the wikipedia package's stock user-agent
# ("Please set a user-agent and respect our robot policy", phab T400119), which surfaced
# as JSONDecodeError on every page()/summary() call while search() kept working. Set a
# descriptive UA and use https. Project-only UA, no personal contact.
import wikipedia.wikipedia as _wp
_wp.API_URL = "https://en.wikipedia.org/w/api.php"
wikipedia.set_user_agent("MAReasoning-eval/0.1 (EPFL NLP research)")
# PATCHED (MAReasoning): Wikimedia's robot policy asks for serialised requests. Without
# this, concurrent solve.py workers burst enough requests to get non-JSON throttle
# responses back, which surfaced as "Error searching Wikipedia: Expecting value:
# line 1 column 1" on ~60% of calls and silently pushed the planner onto its parametric
# fallback instead of retrieval.
from datetime import timedelta as _timedelta
wikipedia.set_rate_limiting(True, min_wait=_timedelta(milliseconds=300))

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "Wikipedia_RAG_Search_Tool"

LIMITATION = f"""
{TOOL_NAME} has the following limitations:
1. It is designed specifically for retrieving grounded information from Wikipedia pages only.
2. Filtering of relevant pages depends on LLM model performance and may not always select optimal pages.
3. The returned information accuracy depends on Wikipedia content quality.
"""

BEST_PRACTICE = f"""
For optimal results with {TOOL_NAME}:
1. Use specific, targeted queries rather than broad or ambiguous questions.
2. The tool automatically filters for relevant pages using LLM-based selection - trust the "relevant_pages" results.
3. If initial results are insufficient, examine the "other_pages" section for additional potentially relevant content.
4. Use this tool as part of a multi-step research process rather than a single source of truth.
5. You can use the {TOOL_NAME} to get more information from the URLs.
"""

class Select_Relevant_Queries(BaseModel):
    matched_queries: list[str]
    matched_query_ids: list[int]

def select_relevant_queries(original_query: str, query_candidates: list[str], llm_engine):

    query_candidates = "\n".join([f"{i}. {query}" for i, query in enumerate(query_candidates)])

    prompt = f"""
You are an expert AI assistant. Your task is to identify and select the most relevant queries from a list of Wikipedia search results that are most likely to address the user’s original question.

## Input

Original Query: `{original_query}`
Query Candidates from Wikipedia Search: `{query_candidates}`

## Instructions

1. Carefully read the original query and the list of query candidates.
2. Select the query candidates that are most relevant to the original query — i.e., those most likely to contain the information needed to answer the question.
3. Return the most relevant queries. If you think multiple queries are helpful, you can return up to 3 queries.
4. Return your output in the following format:

```
- Matched Queries: <list of matched queries>
- Matched Query IDs: <list of matched query ids>. Please make sure the ids are integers. And do not return empty list.
```

## Examples

Original Query: What is the capital of France?
Query Candidates from Wikipedia Search:
0. Closed-ended question
1. France
2. What Is a Nation?
3. Capital city
4. London
5. WhatsApp
6. French Revolution
7. Communes of France
8. Capital punishment
9. Louis XIV

Output:
- Matched Queries: France  
- Matched Query IDs: [1]


Original Query: What is the mass of the moon?
Query Candidates from Wikipedia Search:
0. Moon
1. Planetary-mass moon
2. What If the Moon Didn't Exist
3. Earth mass
4. Moon landing
5. Mass
6. Colonization of the Moon
7. Planetary mass
8. Hollow Moon
9. Gravitation of the Moon

Output:
- Matched Queries: Moon, Planetary-mass moon  
- Matched Query IDs: [0, 1]
"""

    try:
        prompt = prompt.format(original_query=original_query, query_candidates=query_candidates)     

        response = llm_engine.generate(prompt, response_format=Select_Relevant_Queries)
        # print(response)

        # PATCHED (MAReasoning): the vLLM engine accepts response_format but ignores it
        # (engine/vllm.py::_generate_text -> "Chat models without structured outputs"), so
        # `response` is a plain string here and the upstream attribute access raised
        # "'str' object has no attribute 'matched_queries'". That exception was caught
        # below and turned into ([], []), i.e. zero relevant pages -> the tool silently
        # returned nothing but titles. Parse the text format the prompt already asks for,
        # and still accept a real pydantic object if a structured engine is used.
        if hasattr(response, "matched_queries"):
            matched_queries = list(response.matched_queries)
            matched_query_ids = [int(i) for i in response.matched_query_ids]
        else:
            import re as _re
            text = str(response)
            n_candidates = len(query_candidates.strip().split("\n"))
            ids_match = _re.search(
                r"Matched\s+Query\s+IDs?\s*:?\s*\**\s*\[?([0-9,\s]+)\]?", text, _re.I
            )
            matched_query_ids = []
            if ids_match:
                matched_query_ids = [
                    int(tok) for tok in _re.findall(r"\d+", ids_match.group(1))
                ]
            q_match = _re.search(r"Matched\s+Queries\s*:?\s*\**\s*(.+)", text, _re.I)
            matched_queries = (
                [s.strip() for s in q_match.group(1).split(",") if s.strip()]
                if q_match else []
            )
            # Fall back to the top search hit rather than returning nothing at all:
            # an empty list here is indistinguishable from "Wikipedia had no answer",
            # which is exactly the silent failure this patch exists to remove.
            if not matched_query_ids:
                print(
                    "[Wikipedia RAG Search] PATCHED (MAReasoning): could not parse "
                    f"matched query ids from engine output; falling back to top hit. "
                    f"Raw head: {text[:160]!r}"
                )
                matched_query_ids = [0]
            # Guard against hallucinated out-of-range ids (upstream would IndexError).
            matched_query_ids = sorted(
                {i for i in matched_query_ids if 0 <= i < n_candidates}
            )[:3]
        return matched_queries, matched_query_ids
    except Exception as e:
        print(f"Error selecting relevant queries: {e}")
        return [], []

class Wikipedia_Search_Tool(BaseTool):
    def __init__(self, model_string="gpt-4o-mini"):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A tool that searches Wikipedia and returns relevant pages with their page titles, URLs, abstract, and retrieved information based on a given query.",
            tool_version="1.0.0",
            input_types={
                "query": "str - The search query for Wikipedia."
            },
            output_type="dict - A dictionary containing search results, all matching pages with their content, URLs, and metadata.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="What is the exact mass in kg of the moon")',
                    "description": "Search Wikipedia and get the information about the mass of the moon."
                },
                {
                    "command": 'execution = tool.execute(query="Funtion of human kidney")',
                    "description": "Search Wikipedia and get the information about the function of human kidney."
                },
                {
                    "command": 'execution = tool.execute(query="When was the first moon landing?")',
                    "description": "Search Wikipedia and get the information about the first moon landing."
                }
            ],
            user_metadata = {
                "limitation": LIMITATION,
                "best_practice": BEST_PRACTICE
            }
        )
        # PATCHED (MAReasoning): BaseTool.__init__ was called without model_string, so
        # self.model_string stayed None and the Web-RAG handoff in execute() failed
        # silently (bare except -> empty relevant_pages). Set it explicitly.
        self.model_string = model_string
        self.llm_engine = create_llm_engine(model_string=model_string, temperature=0.0, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0)

    def _get_wikipedia_url(self, query):
        """
        Get the Wikipedia URL for a given query.
        """
        query = query.replace(" ", "_") # replace spaces with underscores
        return f"https://en.wikipedia.org/wiki/{query}"

    def search_wikipedia(self, query, max_length=100, max_pages=10, retries=3):
        """
        Search Wikipedia and return candidate pages as {title, url, abstract}.

        PATCHED (MAReasoning): upstream called wikipedia.page(title) for all max_pages=10
        hits to populate `abstract`, i.e. ~20-30 HTTP requests per single tool call. That
        work was wasted twice over: select_relevant_queries() is handed titles only (never
        abstracts), and the Web-RAG step re-fetches the chosen page from its URL anyway.
        The amplification is what triggered Wikimedia throttling under concurrent workers.
        Now: one search request, URLs derived from titles, and page bodies fetched only for
        the pages the selector actually picks. Search is retried with backoff, and a
        throttle/transport failure is reported as an explicit error instead of an empty list.
        """
        import time as _time
        last_err = None
        for attempt in range(retries):
            try:
                search_results = wikipedia.search(query)
                if not search_results:
                    return [{"title": None, "url": None, "abstract": None,
                             "error": f"No results found for query: {query}"}]
                pages_to_process = search_results[:max_pages] if max_pages else search_results
                return [
                    {
                        "title": title,
                        "url": self._get_wikipedia_url(title),
                        "abstract": "Not fetched yet; retrieved_information is filled in for selected pages.",
                    }
                    for title in pages_to_process
                ]
            except Exception as e:
                last_err = e
                if attempt < retries - 1:
                    _time.sleep(1.5 * (2 ** attempt))
        return [{"title": None, "url": None, "abstract": None,
                 "error": f"Error searching Wikipedia after {retries} attempts: {str(last_err)}"}]

    def execute(self, query):
        """
        Searches Wikipedia based on the provided query and returns all matching pages.

        Parameters:
            query (str): The search query for Wikipedia.

        Returns:
            dict: A dictionary containing the search results and all matching pages with their content.
        """
        # PATCHED (MAReasoning): upstream did a hard sys.exit() here when OPENAI_API_KEY
        # was unset, killing the whole solve.py worker rather than degrading. Wikipedia
        # search and page fetch need no key, and the downstream embedding step is now
        # local (see web_search/tool.py::_embed_strings). Gate removed.
            
        # First get relevant queries from the search results
        search_results = self.search_wikipedia(query)

        # Get the titles of the pages   
        titles = [page["title"] for page in search_results if page["title"] is not None]
        if not titles:
            return {"query": query, "relevant_pages": [], "other_pages (may be irrelevant to the query)": search_results}

        # Select the most relevant pages
        matched_queries, matched_query_ids = select_relevant_queries(query, titles, self.llm_engine)
        
        # Only process the most relevant pages
        pages_data = [search_results[i] for i in matched_query_ids]
        other_pages = [search_results[i] for i in range(len(search_results)) if i not in matched_query_ids]

        # For each relevant page, get detailed information using Web RAG
        try:
            from agentflow.tools.web_search.tool import Web_Search_Tool  # PATCHED (MAReasoning): lazy, see header
            web_rag_tool = Web_Search_Tool(model_string=self.model_string)
        except Exception as e:
            print(f"Error creating Web RAG tool: {e}")
            return {"query": query, "relevant_pages": [], "other_pages (may be irrelevant to the query)": search_results}

        for page in pages_data:
            url = page["url"]
            if url is None:
                continue
            try:
                execution = web_rag_tool.execute(query=query, url=url)
                page["retrieved_information"] = execution
            except Exception as e:
                page["retrieved_information"] = None

        return {
            "query": query,
            "relevant_pages (to the query)": pages_data,
            "other_pages (may be irrelevant to the query)": other_pages
        }

    def get_metadata(self):
        """
        Returns the metadata for the Wikipedia_Search_Tool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        metadata = super().get_metadata()
        return metadata


if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:

    cd agentflow/tools/wikipedia_search
    python tool.py
    """

    # Example usage of the Wikipedia_Search_Tool
    tool = Wikipedia_Search_Tool(model_string="gpt-4o-mini")
    # tool = Wikipedia_Search_Tool(model_string="gemini-1.5-flash")
    # tool = Wikipedia_Search_Tool(model_string="dashscope") # 

    # Get tool metadata
    metadata = tool.get_metadata()
    # print(metadata)

    # Sample query for searching Wikipedia
    # query = "Python programming language"
    # query = "what is the main function of the human kidney"
    # query = "What is the mass of the moon"
    # query = "mass of the moon"
    # query = "mass of the moon in kg"
    # query = "What is the mass of the moon (in kg)?"
    # query = "What is the capital of France"
    # query = "Who is Yann LeCun"
    # query = "What is the exact mass in kg of the moon?"
    query = "When was the first moon landing?"

    import json

    # Execute the tool with the sample query
    try:
        # Test with default parameters (all pages)
        execution = tool.execute(query=query)
        print("Execution Result (all pages):")
        print(json.dumps(execution, indent=4))

        # Save the execution result to a JSON file
        os.makedirs("logs", exist_ok=True)
        with open(f"logs/{query}.json", "w") as f:
            json.dump(execution, f, indent=4)
        
    except ValueError as e:
        print(f"Execution failed: {e}")

    print("Done!")
