Project Purpose

You’ve been asked (as a favor to a friend) to analyze a large list of business locations — around 800 rows. Each row corresponds to a physical address, usually tied to a specific company. The friend wants to know:
“What happens at each location?”

Examples of possible answers:

Is this location the company’s headquarters / corporate office?

Is it a manufacturing plant or production facility?

Is it a warehouse / logistics / distribution center / fulfillment center?

Is it an R&D lab, test lab, QA lab, etc.?

Is it a retail storefront or customer service center?

Is it something else, or unclear?

The end goal is a structured dataset (CSV or similar) that, for each address, says:

what kind of site it is,

how confident we are,

and why we think that (evidence / source URLs).

In other words, we’re turning 800 addresses into 800 labeled facility types with audit trails.

Why this isn’t trivial

A company may operate dozens or hundreds of sites. They don’t always clearly label them (“Springfield Facility” could be a distribution hub, or a plastics molding plant, or the headquarters). Sometimes the only public clues are:

job posts (“Production Operator – night shift at our Springfield plant” → manufacturing),

press releases (“new 250,000 sq ft distribution center in Joliet, IL” → warehouse),

permit filings, local news, SEC/EPA docs (often mention “manufacturing facility located at…”),

“About our locations” / “Careers at [City]” pages on the company’s own site,

maps / Google Business listings (“corporate office,” “customer service center,” “R&D lab”).

We’re essentially mining that open information in a systematic way.

Doing this completely by hand for 800 rows is extremely time-consuming and inconsistent. We want to automate it.

High-level approach

The workflow we’re building is:

Take an address from the input CSV.
For now, we assume we have at least an address column. Later we may also include company_name.

Research that address on the public web.
We’ll run targeted search queries like
"<Company Name>" "<Street Address>"
or
"<Company Name>" "<City, State>" facility" "manufacturing" "distribution center".
Then we’ll scrape the relevant pages (company site, press mentions, job listings for that location, etc.) and extract plain text.

Summarize the evidence for that specific location.
We do not want to guess. We want direct statements like:

“The Springfield plant manufactures thermoplastic components for the automotive industry.”

“This 500,000-square-foot distribution center handles e-commerce fulfillment.”

“Regional headquarters for North America sales and customer service.”

We’ll also capture the source URLs for traceability.

Classify the site.
Using that evidence, we classify the site into a discrete label, such as:

Headquarters / admin / office / sales

Manufacturing / production / plant / factory

Warehouse / logistics / distribution / fulfillment

R&D / lab / testing / QA

Retail / storefront / customer-facing service

Other / unclear

Along with:

confidence level (high / medium / low),

short justification,

list of source URLs.

Write results back out.
We produce an output CSV that is basically the input CSV plus new columns:

site_type

confidence

justification (or notes)

source_urls

This final CSV is what your friend ultimately wants.

Why we’re using local LLMs (Ollama)

Instead of sending all of this to a cloud LLM, we’re doing it locally because:

You have a powerful local GPU (RTX 5090).

You already run Ollama and Open WebUI.

You’re comfortable building tooling around local inference.

The work may involve proprietary / internal-ish client data (lists of facilities), so keeping it local is safer.

The project is designed so that local models assist in reasoning and structuring, but you (your code) stay in charge of:

doing searches,

fetching pages,

assembling evidence,

and deciding how results are stored.

The model never “goes online” directly. Your script is the agent.

Where the code fits in

To get there cleanly, we’re building in stages instead of jumping straight to the full autonomous agent.

Stage 1 (what we’re coding right now)

A Python CLI script that:

reads an input CSV of addresses,

calls a local LLM (through Ollama),

and writes an output CSV with extra columns.

This gives us:

argument parsing,

CSV I/O,

a stable interface to Ollama’s /api/generate,

a place to put per-row logic.

Right now the “classification” is just a placeholder prompt that returns dummy JSON. That’s fine — we’re standing up the plumbing first.

Stage 2 (next step)

Enhance the per-row logic so that instead of the dummy output, it:

builds smart search queries for that address,

fetches the top relevant pages,

extracts meaningful text,

asks the model to summarize what this site actually does,

asks the model to choose the correct site type label and confidence.

Stage 3 (polish)

Make the output fully auditable:

Include which URLs we used.

Include short explanation.

Enforce strict JSON from the model so parsing never explodes.

Add rate limiting / caching so we don’t hammer search for the same site twice.

Deal with 800 rows in a batch reliably.

What “success” looks like

When this project is “done enough,” you’ll be able to run something like: