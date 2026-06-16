# AskMyDocs

An agentic document Q&A system where Claude autonomously selects tools to answer questions about your PDFs.

---

## What it does

Upload a PDF, ask a question in plain English. Instead of a fixed pipeline, Claude runs an agent loop — it decides which tools to call, in what order, and when it has enough information to answer.

---

## v1 → v2

**v1** was a fixed RAG pipeline: embed question → retrieve chunks → send to Claude → return answer. Every question followed the same path regardless of what it needed.

**v2** refactors that into a tool-using agent. Claude controls its own flow: it can search the document multiple times with different queries, do arithmetic on what it finds, and optionally analyze figures with vision — all within a single response.

---

## Agent loop

```
User question
      │
      ▼
┌─────────────────────┐
│   Claude (claude-   │
│   sonnet-4-6)       │◄──────────────────────┐
│                     │                       │
│  stop_reason?       │                       │
│  ├─ end_turn ──────►│ return final answer   │
│  └─ tool_use ──────►│ execute tool(s)       │
└─────────────────────┘          │            │
                                 │            │
              ┌──────────────────┤            │
              │                  │            │
              ▼                  ▼            │
      search_documents      calculate         │
      (semantic search      (safe eval)       │
       over PDF chunks)          │            │
              │                  │            │
              └──────────────────┘            │
                       │                      │
                 tool_result ─────────────────┘
```

Claude reasons, calls tools, gets results, reasons again — until it decides the answer is complete. Each iteration is traced to the terminal:

```
[reasoning] I'll search the document for Q3 revenue figures.
[tool] search_documents <- {'query': 'Q3 revenue'}
[reasoning] Now I'll calculate 8% of that figure.
[tool] calculate <- {'expression': '1250000 * 0.08'}
[reasoning] Based on the document, Q3 revenue was $1,250,000...
```

> **Phase 3 screenshot** — add your terminal capture here showing the full reason → search → reason → calculate → answer chain.

---

## Figure analysis (opt-in)

The `analyze_figure` tool sends a page image to Claude's vision model to answer questions about charts, diagrams, or tables that don't survive PDF text extraction.

It is **question-driven, not automatic** — Claude only calls it when the question explicitly targets a visual element. This is intentional:

- **Cost**: each vision call encodes a full page as a base64 image and sends it to a separate model, adding latency and API cost on top of the text search.
- **Latency**: for text questions, vision adds nothing; triggering it unconditionally would make every response slower.

If Claude determines the answer requires a figure, it calls the tool with the page number and a targeted sub-question. Otherwise it stays in the cheaper text path.

---

## Stack

| Layer | Choice |
|---|---|
| UI | Streamlit |
| LLM / agent | Claude (Anthropic API) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers |
| Similarity search | NumPy cosine similarity |
| PDF parsing | PyMuPDF (`fitz`) + pypdf |
| Vision | Claude Sonnet (base64 PNG per page) |

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=your_key_here
```

Run:

```bash
streamlit run app.py
```

---

Built by Rucha Joshi
