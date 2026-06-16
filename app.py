# ── IMPORTS ──────────────────────────────────────────────────────────────
# streamlit turns this Python script into a web app — no HTML needed
import streamlit as st
# anthropic is the official SDK to talk to Claude API
import anthropic
# os lets us read environment variables like our API key
import os
import base64
import tempfile
# numpy is a math library — we use it for vector similarity calculations
import numpy as np
# pypdf reads and extracts text from PDF files
from pypdf import PdfReader
# sentence_transformers converts text into numerical vectors (embeddings)
# all-MiniLM-L6-v2 is a small, fast, free model — perfect for this use case
from sentence_transformers import SentenceTransformer
# dotenv reads our .env file so our API key stays secret
from dotenv import load_dotenv
import fitz

# ── SETUP ─────────────────────────────────────────────────────────────────
# load the .env file so os.getenv() can find ANTHROPIC_API_KEY
load_dotenv()

# create the Claude client — our connection to the Anthropic API
# os.getenv() reads the key from .env so we never hardcode it
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# load the sentence embedding model
# this runs once when the app starts — Streamlit caches it automatically
# it downloads the model on first run (~80MB), then uses the cached version
embedder = SentenceTransformer('all-MiniLM-L6-v2')


tools = [
    {
        "name": "search_documents",
        "description": "Search the uploaded PDF text for passages relevant to a query using semantic vector search. Use this whenever the answer might be in the document's text. You can call it multiple times with different queries to gather more context.",
        "input_schema":{
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
# ── FUNCTION 1: LOAD AND CHUNK PDF ───────────────────────────────────────
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

    # create overlapping chunks of ~500 words
    # the step is chunk_size - 50 = 450, meaning consecutive chunks
    # share 50 words of overlap — this prevents losing context at boundaries
    # example: if a sentence spans the end of chunk 1 and start of chunk 2,
    # the overlap ensures Claude sees the full sentence in at least one chunk
    chunks = []
    for i in range(0, len(words), chunk_size - 50):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:  # skip any empty chunks
            chunks.append(chunk)

    # returns a list of text strings — e.g. ["first 500 words...", "next 500 words..."]
    return chunks


# ── FUNCTION 2: BUILD VECTOR INDEX ───────────────────────────────────────
def build_index(chunks):
    # convert all text chunks into embedding vectors using sentence-transformers
    # embedder.encode() returns a 2D numpy array: shape = (num_chunks, 384)
    # 384 is the vector dimension of the all-MiniLM-L6-v2 model
    # each chunk becomes a list of 384 numbers representing its meaning
    embeddings = embedder.encode(chunks)

    # store both the original text chunks AND their embeddings together
    # we need chunks to show the user the source text
    # we need embeddings to do similarity search at query time
    return {"chunks": chunks, "embeddings": embeddings}


# search_documents = It takes the same `index` dict that build_index() returns
# and returns the joined context string — it does NOT call Claude.
def search_documents(query: str, index: dict) -> str:
    q_embedding = embedder.encode([query])[0]
    embeddings = index["embeddings"]
    chunks = index["chunks"]

    # identical cosine-similarity math from your answer_question()
    similarities = np.dot(embeddings, q_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_embedding) + 1e-10
    )
    top_3 = np.argsort(similarities)[-3:][::-1]
    top_chunks = [chunks[i] for i in top_3]

    return "\n\n".join(top_chunks)

def calculate(expression: str) -> str:
    allowed = {"__builtins__": {}}
    try:
        return str(eval(expression, allowed, {}))   # note in README you'd harden this
    except Exception as e:
        return f"Error: {e}"

def analyze_figure(page_number: int, question: str, page_images: dict) -> str:
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

# The dispatcher takes `index` so search_documents can reach the embeddings.
# `index` is built once at upload via load_and_chunk_pdf() -> build_index().
def execute_tool(name: str, tool_input: dict, index: dict, page_images: dict[int, str]) -> str:
    if name == "search_documents":
        return search_documents(tool_input["query"], index)
    if name == "calculate":
        return calculate(tool_input["expression"])
    if name == "analyze_figure":
        return analyze_figure(tool_input["page_number"], tool_input["question"], page_images)
    return f"Unknown tool: {name}"


SYSTEM_PROMPT = (
    "You are a document analyst with access to four tools: "
    "search_documents (semantic search over the uploaded PDF), "
    "calculate (arithmetic), "
    "web_search (live internet search — use this whenever the question asks about current events, "
    "real-time data, recent figures, or anything that may have changed after the document was written), "
    "and analyze_figure (vision analysis of a specific page — use only for questions about charts, "
    "graphs, or images that text search cannot answer). "
    "Always search the document first. If the document does not contain the answer and the question "
    "involves current or recent information, use web_search."
)

def run_agent(user_question: str, index: dict, page_images: dict, is_scanned: bool = False) -> str:
    system = SYSTEM_PROMPT
    if is_scanned:
        system += (
            " IMPORTANT: This PDF is a scanned document with no extractable text. "
            "The search_documents tool will not return useful content — skip it and go straight to "
            "analyze_figure to read the relevant page(s) visually."
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
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"[reasoning] {block.text}")
            if block.type == "tool_use":
                print(f"[tool] {block.name} <- {block.input}")

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        # Claude wants tools — append its turn FIRST
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input, index, page_images)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,    # MUST match the id from the call
                    "content": result,
                })

        messages.append({"role": "user", "content": tool_results})

def render_pdf_pages(pdf_path: str) -> dict[int, str]:
    out_dir = tempfile.mkdtemp()
    doc = fitz.open(pdf_path)
    page_images = {}
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=100)
        # if an unusually large page still exceeds ~1500px wide, scale it down
        if pix.width > 1500:
            pix = page.get_pixmap(dpi=int(100 * 1500 / pix.width))
        path = f"{out_dir}/page_{i}.jpg"
        pix.save(path)
        page_images[i] = path
    return page_images


# ── STREAMLIT PAGE CONFIG ─────────────────────────────────────────────────
# must be the first Streamlit command — sets browser tab title and layout
st.set_page_config(page_title="AskMyDocs", page_icon="📄", layout="wide")

# ── CUSTOM CSS ────────────────────────────────────────────────────────────
# st.markdown with unsafe_allow_html=True lets us inject raw CSS
# this styles the app with a dark purple gradient theme
st.markdown("""
<style>
    /* dark purple gradient background */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    /* make all text white so it's readable on dark background */
    .stApp, .stApp p, .stApp label, .stMarkdown { color: white !important; }
    /* purple title */
    h1 { color: #a78bfa !important; font-size: 3rem !important; }
    /* slightly darker purple for subheadings */
    h2, h3 { color: #7c3aed !important; }
    /* styled file upload box with dashed purple border */
    .stFileUploader { background: rgba(255,255,255,0.05); border: 2px dashed #7c3aed; border-radius: 12px; padding: 1rem; }
    /* dark chat input with purple border */
    .stChatInput input { background: rgba(255,255,255,0.1) !important; border: 1px solid #7c3aed !important; color: white !important; border-radius: 20px !important; }
    /* purple divider line */
    hr { border-color: #7c3aed !important; opacity: 0.3; }
    /* light purple caption text */
    .stCaption { color: #a78bfa !important; }
</style>
""", unsafe_allow_html=True)


# render the app header
st.title("AskMyDocs")
st.caption("Upload a PDF · Ask questions in plain English · Powered by Claude · Built by Rucha Joshi")
st.divider()


# ── SESSION STATE ─────────────────────────────────────────────────────────
# Streamlit re-runs the ENTIRE script from top to bottom on every interaction
# st.session_state is a dictionary that persists across re-runs
# without session_state, the index and chat history would reset on every click

# index = our vector index {"chunks": [...], "embeddings": [...]}
# starts as None — gets populated after a PDF is uploaded
if "index" not in st.session_state:
    st.session_state.index = None

# messages = full chat history as a list of {"role": "user/assistant", "content": "..."}
# starts empty — grows as the user asks questions
if "messages" not in st.session_state:
    st.session_state.messages = []

# doc_name = filename of currently loaded PDF
# used to detect when the user uploads a DIFFERENT document
# prevents re-indexing the same document on every button click
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

if "page_images" not in st.session_state:
    st.session_state.page_images = {}

if "is_scanned" not in st.session_state:
    st.session_state.is_scanned = False

if "upload_status" not in st.session_state:
    st.session_state.upload_status = None  # ("success"|"warning", message)


# ── FILE UPLOAD ───────────────────────────────────────────────────────────
st.markdown("**Step 1 — Upload your PDF** (text-based or scanned)")
uploaded_file = st.file_uploader(
    "Drag and drop a PDF here, or click Browse files",
    type=["pdf"],
    help="Supports text PDFs and scanned documents. Scanned PDFs will use visual page analysis instead of text search.",
)

if uploaded_file and uploaded_file.name != st.session_state.doc_name:
    with st.spinner(f"Reading {uploaded_file.name}..."):
        chunks = load_and_chunk_pdf(uploaded_file)
        is_scanned = len(chunks) == 0
        st.session_state.is_scanned = is_scanned

        if is_scanned:
            # build a minimal index so the agent loop still works;
            # search_documents will return this message and Claude will pivot to analyze_figure
            chunks = ["[This PDF contains no extractable text — it is a scanned document. "
                      "Use the analyze_figure tool to read specific pages visually.]"]

        st.session_state.index = build_index(chunks)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        st.session_state.page_images = render_pdf_pages(tmp_path)
        st.session_state.doc_name = uploaded_file.name
        st.session_state.messages = []

        if is_scanned:
            st.session_state.upload_status = (
                "warning",
                f"**{uploaded_file.name}** loaded ({len(st.session_state.page_images)} pages). "
                "No text layer detected — this looks like a scanned PDF. "
                "Ask questions about specific pages or figures; Claude will analyze them visually.",
            )
        else:
            st.session_state.upload_status = (
                "success",
                f"**{uploaded_file.name}** ready — {len(chunks)} sections indexed. "
                "Step 2: type your question below.",
            )

# show persistent status so it survives re-runs triggered by the chat input
if st.session_state.upload_status:
    kind, msg = st.session_state.upload_status
    if kind == "success":
        st.success(msg)
    else:
        st.warning(msg)


# ── CHAT INTERFACE ────────────────────────────────────────────────────────
if st.session_state.index:
    st.markdown("**Step 2 — Ask a question**")
    st.subheader("Chat with your document")

    # replay the full chat history so the conversation stays visible
    # st.chat_message() creates a chat bubble with the role's avatar
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):  # "user" or "assistant"
            st.write(msg["content"])

    # render the chat input box — returns the message when user presses Enter
    # returns None when no message has been typed yet
    question = st.chat_input("Ask a question about your document...")

    if question:
        # immediately show the user's message in the chat
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # generate and display Claude's answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # call our RAG pipeline:
                    # 1. embed the question
                    # 2. find top 3 similar chunks
                    # 3. send chunks + question to Claude
                    # 4. return Claude's answer + source chunks
                    answer = run_agent(
                        question,
                        st.session_state.index,
                        st.session_state.page_images,
                        is_scanned=st.session_state.is_scanned,
                    )

                    # display Claude's answer as the assistant's message
                    # escape $ so Streamlit doesn't interpret dollar amounts as LaTeX math
                    st.markdown(answer.replace("$", r"\$"))

                    # save Claude's answer to chat history for display on next re-run
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    # catch any error (API failure, bad PDF, etc.)
                    # show friendly message instead of crashing the app
                    st.error(f"Error: {e}")

else:
    st.info("Upload a PDF above (Step 1) to get started. The chat will appear here once your document is ready.")