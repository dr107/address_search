Project Status (October 2025)
--------------------------------

You are building an address-enrichment pipeline that labels ~800 business locations with site type, confidence, and supporting evidence. The system now has a robust CLI scaffold, optional web-research plumbing, and a conditional agentic loop for tool-capable LLMs. This document captures the current state so you can quickly resume development on your main workstation.


High-Level Flow
---------------

1. `main.py` handles CLI parsing (input/output paths, Ollama model/URL, logging level, row limits) and wires configuration to the processing pipeline.  
2. `csvparse.process_file`  
   - loads the CSV (`Full Address` + `Company Name` columns expected),  
   - determines whether to engage web research and/or the agentic tool loop,  
   - iterates rows, calling `classification.run_model_on_address`,  
   - writes the enriched CSV (adds `site_type`, `confidence`, `notes`, `raw_model_output`).  
3. `classification.py` either:
   - runs the **tool agent** (LLM can call `web_search`/`fetch_url` via Ollama `/api/chat`) and parses the final JSON, or  
   - falls back to a **single-shot prompt** enriched with any evidence gathered by `research.py`.  
4. `research.py` houses the DuckDuckGo OpenAPI client (POST to your local MCP server at `http://localhost:8001` by default), the HTML fetcher, and a `ResearchPipeline` that builds queries and returns `EvidenceDocument`s.  
5. `agentic.py` defines the tool loop and abstracts the two tools. It automatically exits—and `classification.py` falls back—if the model/runtime cannot service `/api/chat` (e.g., current `llama3` build on M1).  


CLI Usage Cheatsheet
--------------------

```
source .venv/bin/activate
python main.py \
  --input data/nj_companies.csv \
  --output data/short_out.csv \
  --limit 10 \
  --model llama3.1-8b-instruct \
  --enable-web-research \
  --agentic-mode auto \
  --log-level DEBUG
```

Key flags:
- `--enable-web-research` – engages the pre-agent research pipeline (DuckDuckGo API + page fetch) and passes the evidence into the single-shot prompt.  
- `--agentic-mode {auto,on,off}` – controls the tool loop. `auto` checks `--model` against `--agentic-model-hints` (defaults include `llama3`, `llama4`, `deepseek`). If a compatible model is detected and Ollama’s `/api/chat` works, the agent drives its own search/fetch calls. Otherwise the code falls back gracefully and logs a warning.  
- `--search-api-url` – base URL for your DuckDuckGo MCP server (`http://localhost:8001`).  
- `--max-search-results`, `--max-documents`, `--fetch-timeout` – tune the research intensity.  
- `--agent-max-iterations` – caps the number of tool-call turns (default 6).  
- `--limit` – helpful for dry runs.  
- `--log-level DEBUG` – recommended when testing agent mode to see tool requests/responses and any JSON recovery.


Current Capabilities & Behavior
-------------------------------

* **Input expectations** – CSV must provide `Company Name` and `Full Address`. Missing addresses log a warning and produce empty rows.  
* **Ollama integration** – All inference is local. The single-shot path hits `/api/generate`; the agent path uses `/api/chat`. If `/api/chat` returns 400 (current behavior on M1 llama3), the system logs the issue and reruns the request in single-shot mode.  
* **Evidence handling** – When web research is enabled (but agent mode isn’t), the pipeline searches DuckDuckGo via the OpenAPI proxy, fetches up to `max_documents` pages, and injects trimmed text into the prompt.  
* **JSON robustness** – `_parse_model_response` now attempts to salvage JSON even if the model wraps it with narration (it scans the response for the first `{...}` block). If no valid JSON is found, `notes` explains the failure and the raw output is preserved for debugging.  
* **Logging** – Rich INFO/DEBUG logs track client initialization, row processing, agent loop iterations, search queries, and JSON parsing issues. Use `--log-level DEBUG` whenever you want to inspect the raw evidence flow.  
* **Dependencies** – `requests` and `beautifulsoup4` are required (see `requirements.txt`). The `.venv` you created already has them installed.  


Known Limitations / Future Work
-------------------------------

1. **Agentic loop compatibility** – The current M1 llama3 build in Ollama does not support `/api/chat`, so the agent is effectively disabled (falls back every time). On your main RTX box with newer Llama4/Deepseek builds, the loop should work; if not, tweak `--agentic-model-hints` or upgrade Ollama.  
2. **Search quality** – DuckDuckGo via MCP is functional but simplistic. Consider swapping in a richer search API (Brave Search, Bing, SerpAPI) once you’re ready for Stage 2/3 work. The `DuckDuckGoAPIClient` interface makes this a drop-in change.  
3. **Evidence aggregation** – Currently we just slice the page text. Future steps include smarter HTML cleaning, source ranking, and capturing explicit citations/URLs in the final CSV.  
4. **JSON schema enforcement** – We still ask the model for a simple object. Eventually we’ll want a stricter schema (site_type enum, confidence enum, array of sources) plus validation.  
5. **Scalability** – No batching, retry, or long-run resilience yet (e.g., rate limiting, caching). Before running all 800 rows, add checkpoints and maybe a resume mechanism.  
6. **Model selection** – README previously referenced local LLMs generically. Final plan is to use stronger models such as Deepseek-R1-70B or Llama 4 variants on the workstation GPU(s). When you switch models, update the CLI defaults or pass `--model` explicitly.  


Quick “When You Return” Checklist
---------------------------------

1. Pull latest code and confirm your workstation has `requests`, `beautifulsoup4`, and any other dependencies installed.  
2. Ensure the DuckDuckGo MCP server is running at `http://localhost:8001`.  
3. Test a small batch with the target high-power model, enabling agent mode (`--agentic-mode on --log-level DEBUG`) to verify tool calls.  
4. Decide whether to run with the external research pipeline, pure agent mode, or both (agent mode currently bypasses the pipeline to avoid duplicated searches).  
5. Plan Stage 2 work: better query templates, evidence structuring, and schema validation.  

With this snapshot, you should be able to step back into the project, know what’s working, what’s intentionally stubbed, and where to push next. Happy hacking!
