"""
askmydocs_core.py — the AskMyDocs retrieval + agent pipeline.

This module is the single source of truth for the pipeline logic. Both the
Streamlit UI (app.py) and the eval harness (testset.py) import from here, so
the harness measures the *real* pipeline rather than a re-implementation.

It contains NO Streamlit code, so it is safe to `import askmydocs_core` from a
plain Python process.
"""

import os
import base64
import tempfile
from pathlib import Path

import numpy as np
import anthropic
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# PyMuPDF exposes itself as both `pymupdf` (canonical) and `fitz` (legacy).
# Prefer `pymupdf` — the `fitz` name can be shadowed by an unrelated package.
try:
    import pymupdf as fitz
except ImportError:
    import fitz

# Load .env from next to this file, regardless of the launch directory.
load_dotenv(Path(__file__).with_name(".env"))

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "No Anthropic API key found. Add a `.env` file next to askmydocs_core.py "
        "containing `ANTHROPIC_API_KEY=sk-ant-...`, or set the environment variable."
    )

client = anthropic.Anthropic(api_key=API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")


tools = [
    {
        "name": "search_documents",
        "description": "Search the uploaded PDF text for passages relevant to a query using semantic vector search. Use this whenever the answer might be in the document's text. You can call it multiple times with different queries to gather more context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query to find relevant document passages"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Use this for any arithmetic rather than doing math yourself.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "A math expression, e.g. '1250 * 0.08'"}
            },
            "required": ["expression"]
        }
    },
    {
        "type": "web_search_20260209",
        "name": "web_search"
    },
    {
        "name": "analyze_figure",
        "description": "Visually analyze a specific page of the document to interpret charts, figures, diagrams, tables, or images. ONLY use this when the user's question is about a visual element (a chart, graph, figure, diagram, or image) that text search cannot answer. Do not use it for ordinary text questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "page_number": {"type": "integer", "description": "The 1-based page number containing the figure to analyze"},
                "question": {"type": "string", "description": "What to determine from the figure, e.g. 'What trend does this chart show?'"}
            },
            "required": ["page_number", "question"]
        }
    }
]


def load_and_chunk_pdf(uploaded_file, chunk_size=500):
    reader = PdfReader(uploaded_file)
    full_text = " ".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )

    # PyMuPDF fallback — handles compressed/malformed text streams that pypdf misses
    if not full_text.strip():
        doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
        full_text = " ".join(
            page.get_text() for page in doc if page.get_text().strip()
        )

    words = full_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 50):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def build_index(chunks):
    embeddings = embedder.encode(chunks)
    return {"chunks": chunks, "embeddings": embeddings}


def search_documents(query: str, index: dict) -> str:
    q_embedding = embedder.encode([query])[0]
    embeddings = index["embeddings"]
    chunks = index["chunks"]
    similarities = np.dot(embeddings, q_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_embedding) + 1e-10
    )
    top_3 = np.argsort(similarities)[-3:][::-1]
    return "\n\n".join(chunks[i] for i in top_3)


def calculate(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


def analyze_figure(page_number: int, question: str, page_images: dict) -> str:
    if page_number not in page_images:
        valid = sorted(page_images.keys())
        if not valid:
            return "This document has no page images available to analyze."
        return (f"Page {page_number} does not exist in this document. "
                f"Valid page numbers are {valid[0]} to {valid[-1]}. Please try again with a valid page.")
    img_path = page_images[page_number]
    with open(img_path, "rb") as f:
        img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
    media_type = "image/jpeg" if img_path.endswith(".jpg") else "image/png"
    vision_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
                {"type": "text", "text": f"This is page {page_number} of a document. {question}"}
            ]
        }]
    )
    return "".join(b.text for b in vision_response.content if b.type == "text")


def execute_tool(name: str, tool_input: dict, index: dict, page_images: dict) -> str:
    try:
        if name == "search_documents":
            return search_documents(tool_input["query"], index)
        if name == "calculate":
            return calculate(tool_input["expression"])
        if name == "analyze_figure":
            return analyze_figure(tool_input["page_number"], tool_input["question"], page_images)
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool '{name}' encountered an error: {type(e).__name__}: {e}"


SYSTEM_PROMPT = (
    "You are a document analyst with access to four tools:\n"
    "- search_documents: semantic search over the uploaded PDF. Always try this first for "
    "questions about the document's own content.\n"
    "- calculate: arithmetic. Use it instead of doing math yourself.\n"
    "- web_search: LIVE internet search. You MUST call web_search — never answer from your own "
    "training knowledge — whenever the question involves any of the following:\n"
    "    • current events, recent news, or 'latest'/'current'/'today' phrasing;\n"
    "    • real-time or recent figures (prices, costs, rates, statistics) that change over time;\n"
    "    • comparing a value in the document to a benchmark, industry average, 'typical' range, "
    "or what is 'normal' — these require up-to-date external data, not your prior knowledge;\n"
    "    • anything that may have changed after the document was written.\n"
    "  If you find yourself about to state an industry figure, average, or benchmark from memory, "
    "STOP and call web_search instead. A benchmark answered from training data is a failure.\n"
    "- analyze_figure: vision analysis of a specific page. Use only for charts, graphs, or images "
    "that text search cannot answer.\n"
    "When you cite a figure obtained from web_search, briefly note that it came from a live search."
)


def run_agent(user_question: str, index: dict, page_images: dict, documents: list = None):
    """Returns (answer_text, trace) where trace is a list of human-readable step strings."""
    trace = []
    page_count = len(page_images)
    documents = documents or []

    if len(documents) <= 1:
        system = SYSTEM_PROMPT + (
            f" The document has {page_count} page(s); "
            f"valid page numbers for analyze_figure are 1 to {page_count}."
        )
    else:
        doc_info = ", ".join(
            f"'{d['name']}' (pages {d['page_start']}–{d['page_start'] + d['page_count'] - 1})"
            for d in documents
        )
        system = SYSTEM_PROMPT + (
            f" {len(documents)} documents are loaded: {doc_info}. "
            f"Total pages 1–{page_count}. "
            "Each search result is prefixed with [Source: filename] so you can tell which document it comes from. "
            f"Valid page numbers for analyze_figure are 1 to {page_count}."
        )

    scanned_docs = [d["name"] for d in documents if d["is_scanned"]]
    if scanned_docs:
        system += (
            f" NOTE: {', '.join(scanned_docs)} are scanned documents with no extractable text — "
            "skip search_documents for questions about those and use analyze_figure on their pages directly."
        )
    messages = [{"role": "user", "content": user_question}]
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system,
            tools=tools,
            messages=messages
        )

        # ── Trace every block, including SERVER-side tools (web_search) ──
        for block in response.content:
            if block.type == "text" and block.text.strip():
                line = f"💭 {block.text.strip()}"
                print(f"[reasoning] {block.text}")
            elif block.type == "tool_use":
                line = f"🔧 {block.name}({block.input})"
                print(f"[tool] {block.name} <- {block.input}")
            elif block.type == "server_tool_use":
                # web_search runs server-side and surfaces as server_tool_use
                line = f"🌐 web_search({block.input.get('query', block.input)})"
                print(f"[server_tool] {block.name} <- {block.input}")
            elif block.type == "web_search_tool_result":
                n = len(block.content) if isinstance(block.content, list) else "?"
                line = f"🌐 web_search returned {n} results"
                print(f"[server_tool_result] {n} results")
            else:
                continue
            trace.append(line)

        # Server-side tool loop hit its iteration cap — re-send to resume.
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue

        # No client tool requested → we're done (server tools already resolved).
        if response.stop_reason != "tool_use":
            answer = "".join(b.text for b in response.content if b.type == "text")
            return answer, trace

        # Client tools requested — execute them and feed results back.
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input, index, page_images)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
        messages.append({"role": "user", "content": tool_results})


def render_pdf_pages(pdf_path: str) -> dict:
    out_dir = tempfile.mkdtemp()
    doc = fitz.open(pdf_path)
    page_images = {}
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=100)
        if pix.width > 1500:
            pix = page.get_pixmap(dpi=int(100 * 1500 / pix.width))
        path = f"{out_dir}/page_{i}.jpg"
        pix.save(path)
        page_images[i] = path
    return page_images
