**Take-home: Company Leadership RAG Agent
Input. A company's domain or website URL.
Goal. Build an end-to-end system that (a) finds the company's leadership from public sources
and (b) exposes a chat interface to ask questions over the collected data.
Scope of "leadership.
" C-level executives (CEO, CTO, CFO, CMO, …), Vice Presidents, and
Heads of departments.
Two parts.
1. Data collection & storage — find the people, collect their public profile data, decide on
a storage model (your call: flat files, SQLite, vector store, graph — whatever fits).
2. Chat interface — an LLM-powered chat over the dataset that handles questions like
"Who's their CTO?"
,
"How many VPs do they have?",
"Who heads marketing?"
"Where , is their CEO based?"
.
Rules.
Real LLMs only — no stubs or mocked completions.
No technical constraints on stack, libraries, or architecture. Lean / no-overengineering is
welcomed.
You must use any coding agent for development.
Deliverables.

Runnable Git repository (we should be able to clone-and-go).
Data fixtures — the dataset you collected for the example company.
Chat interface in any reasonable form (CLI, web, notebook, Claude/ChatgGPT app —
your choice).
session.json or chat export from your coding assistant session so we can see how
you worked with the agent.
Test Inputs.
-
-
https://meetcampfire.com/
https://robinhood.com/
Effort & deadline. Estimated 4–8 hours of actual work. No hard deadline — one to two weeks
is fine; tell us when you'll deliver.
What we're evaluating. How you approach (1) data sourcing, (2) storage design, (3) agent
architecture, (4) your collaboration loop with Claude Code.**