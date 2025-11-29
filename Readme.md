Project Status (November 2025)
--------------------------------

This project began as a request from a friend. He's a chemical engineer and is looking to relocate to another state. To find potential employers, he reached out to a regulatory agency in NJ and asked for a list of chemical plants. They gave him a list of ~800 names and addresses. However, not all of these businesses are chemical plants, some are just administrative offices, etc, and some aren't even open anymore. 

My friend asked my help getting ChatGPT to help him categorize these addresses, but it was pretty clear to me that ChatGPT isn't going to do 800 rows of agentic research for you, not even on the paid plan. So I used the Cursor IDE and OpenAI Codex to throw this together. The current state (Nov '25) represents maybe ~4h of active effort. 

Here's roughly what the app does
- It uses python's `csv` library to read the input CSV.
- For each row, it takes the name and address and does a couple different web searches to see what comes up
- It fetches a number of pages in those results, and sends them to a local LLM to summarize, and attempt to categorize the plant into one of a few categories (which the caller can provide as input)

Next Steps
--------------
- Refactor code to be more in line with my stylistic choices
- Attempt to tune accuracy/price ratio by playing with the amount of results processed, and perhaps using the LLM to "judge" results as useful or not before including them in the final context window for summarization.
- Generify. Remove all direct references to chemicals and such, and make this run as a generic business address classifier.
- Make it run faster and with fewer dependencies by replacing Ollama with VLLM or another lower-level framework, perhaps enabling more efficient usage of the model
- Fully containerize the setup so that it can be installed in one command.

Tech stack
---------------
The tech stack involves
- Ollama API to provide a uniform interface for local LLM inference
- EXA search for search and page fetch features (I tried duckduckgo for search and was rate limited, and getting sites to respond to my page fetches without 40X was too frustrating)
- Python to glue it all together.

High-Level Flow
---------------

1. `main.py` handles CLI parsing (input/output paths, Ollama model/URL, logging level, row limits) and wires configuration to the processing pipeline. Default inference model is now `llama3.1:70b-instruct-q4_0`.  
2. `csvparse.process_file`  
   - loads the CSV (`Full Address` + `Company Name` columns expected) and optionally hydrates an **output-cache** from the previous run (skipped via `--ignore-cache`),  
   - determines whether to engage web research and/or the agentic tool loop,  
   - orchestrates the **planning ➞ research ➞ summarization ➞ classification** pipeline for each uncached row,  
   - writes the enriched CSV (adds `site_type`, `confidence`, `notes`, `raw_model_output`, `query_plan`, `evidence_summary`, `category_suggestions`).  
3. `classification.py` either:
   - runs the **tool agent** (LLM can call `web_search`/`fetch_url` via Ollama `/api/chat`) and parses the final JSON, or  
   - falls back to a **single-shot prompt** enriched with any evidence gathered by `research.py`.  
4. `research.py` houses the Exa SDK wrapper for search + content retrieval and a `ResearchPipeline` that now consumes a `QueryPlan` before collecting `EvidenceDocument`s.  
5. `agentic.py` defines the tool loop and abstracts the two tools. It automatically exits—and `classification.py` falls back—if the model/runtime cannot service `/api/chat` (e.g., current `llama3` build on M1).  
6. `planning.py` (LLM-backed query planner) and `summarizer.py` (evidence-to-bullets summarizer) provide the new intermediate stages so the final classifier sees a compact rationale along with raw excerpts.  


CLI Usage Cheatsheet
--------------------

```
source venv/bin/activate
python main.py \
  --input data/nj_companies.csv \
  --output data/short_out.csv \
  --limit 10 \
  --model "llama3.1:70b-instruct-q4_0" \
  --enable-web-research \
  --agentic-mode auto \
  --log-level DEBUG
```

Key flags:
- `--enable-web-research` – engages the pre-agent research pipeline (Exa search + content retrieval) and passes the evidence into the single-shot prompt.  
- `--agentic-mode {auto,on,off}` – controls the tool loop. `auto` checks `--model` against `--agentic-model-hints` (defaults include `llama3`, `llama4`, `deepseek`). If a compatible model is detected and Ollama’s `/api/chat` works, the agent drives its own search/fetch calls. Otherwise the code falls back gracefully and logs a warning.  
- `--max-search-results` / `--exa-results`, `--max-documents` – tune the research intensity.  
- `--fetch-timeout` – retained for compatibility; Exa manages its own request timeouts.  
- `--random-sample` – when paired with `--limit`, randomly choose which rows to process instead of taking the first N.  
- `--expanded-search-results`, `--expanded-max-documents` – override the secondary, deeper evidence pass that triggers when a row remains inconclusive.  
- `--agent-max-iterations` – caps the number of tool-call turns (default 6).  
- `--limit` – helpful for dry runs.  
- `--log-level DEBUG` – recommended when testing agent mode to see tool requests/responses and any JSON recovery.
- `--ignore-cache` – forces every row to recompute even if the target output CSV already contains a prior result. Leave unset for automatic row-level caching/resume behavior.
- `--categories-file` – path to a JSON file describing category options. File format: a list (or `{ "categories": [...] }`) of objects with `name` and optional `description`. Example: `categories.sample.json`. If omitted, the model invents categories on the fly; when provided, both the single-shot and agentic prompts prefer your named options and the `category_suggestions` column records which guidance was used.
- `--batch-size` – number of rows to process before persisting progress (default 5). Each batch rewrite updates the output CSV so long runs can resume with minimal loss if interrupted.

Categories file example (`categories.sample.json`):

```
[
  {"name": "manufacturing", "description": "Produces goods or pharmaceuticals on-site"},
  {"name": "distribution", "description": "Warehousing, logistics, or order fulfillment"},
  {"name": "office", "description": "Corporate or administrative offices"}
]
```


Current Capabilities & Behavior
-------------------------------

* **Input expectations** – CSV must provide `Company Name` and `Full Address`. Missing addresses log a warning and produce empty rows. Optional categories guidance (via `--categories-file`) is stored in the `category_suggestions` column so cache hits remain accurate when you change the hints.  
* **Ollama integration** – All inference is local. The single-shot path hits `/api/generate`; the agent path uses `/api/chat`. If `/api/chat` returns 400 (current behavior on M1 llama3), the system logs the issue and reruns the request in single-shot mode.  
* **Evidence handling** – When web research is enabled, the pipeline now: (a) asks the planner for targeted queries, (b) runs an initial low-impact Exa search that just combines company + address, (c) fetches candidate pages concurrently via Exa, (d) summarizes the harvested docs into a short rationale, (e) feeds both the summary and raw evidence into the classifier/agent prompts, and (f) automatically performs a deeper follow-up search—with additional heuristic queries and higher limits—if the first pass leaves the classification inconclusive.  
* **Row-level cache** – If the destination CSV already exists, its rows are keyed by `(Company Name, Full Address)` and reused automatically so reruns can pick up where they left off. Pass `--ignore-cache` to recompute from scratch.  
* **JSON robustness** – `_parse_model_response` now attempts to salvage JSON even if the model wraps it with narration (it scans the response for the first `{...}` block). If no valid JSON is found, `notes` explains the failure and the raw output is preserved for debugging.  
* **Logging** – Rich INFO/DEBUG logs track client initialization, row processing, agent loop iterations, search queries, and JSON parsing issues. Use `--log-level DEBUG` whenever you want to inspect the raw evidence flow.  
* **Dependencies** – `requests` and `exa-py` are required (see `requirements.txt`). The `.venv` you created already has them installed.  


Known Limitations / Future Work
-------------------------------

1. **Agentic loop compatibility** – The current M1 llama3 build in Ollama does not support `/api/chat`, so the agent is effectively disabled (falls back every time). On your main RTX box with newer Llama4/Deepseek builds, the loop should work; if not, tweak `--agentic-model-hints` or upgrade Ollama.  
2. **Search quality / rate limits** – Exa yields strong results but enforces throughput caps. Monitor usage and tune `--exa-results`/`--max-documents` if you hit throttling. The `ExaSearchClient` wrapper keeps the integration localized should you need to swap providers later.  
3. **Evidence aggregation** – Currently we just slice the page text. Future steps include smarter HTML cleaning, source ranking, and capturing explicit citations/URLs in the final CSV.  
4. **JSON schema enforcement** – We still ask the model for a simple object. Eventually we’ll want a stricter schema (site_type enum, confidence enum, array of sources) plus validation.  
5. **Scalability** – No batching, retry, or long-run resilience yet (e.g., rate limiting, smarter cache invalidation). Before running all 800 rows, add checkpoints and maybe a resume mechanism. Parallel web fetches improve latency, but there’s still no backoff logic if many sites block requests.  
6. **Model selection** – README previously referenced local LLMs generically. Final plan is to use stronger models such as Deepseek-R1-70B or Llama 4 variants on the workstation GPU(s). When you switch models, update the CLI defaults or pass `--model` explicitly.  


Quick “When You Return” Checklist
---------------------------------

1. Pull latest code and confirm your workstation has `requests`, `exa-py`, and any other dependencies installed.  
2. Export a valid `EXA_TOKEN` in your shell before running the pipeline.  
3. Test a small batch with the target high-power model, enabling agent mode (`--agentic-mode on --log-level DEBUG`) to verify tool calls.  
4. Decide whether to run with the external research pipeline, pure agent mode, or both (agent mode currently bypasses the pipeline to avoid duplicated searches).  
5. Plan Stage 2 work: better query templates, evidence structuring, and schema validation.  

With this snapshot, you should be able to step back into the project, know what’s working, what’s intentionally stubbed, and where to push next. Happy hacking!
