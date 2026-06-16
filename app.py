import streamlit as st
import anthropic
import os
import base64
import tempfile
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import fitz

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embedder = SentenceTransformer('all-MiniLM-L6-v2')


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


def render_pdf_pages(pdf_path: str) -> dict[int, str]:
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


# ── PAGE CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(page_title="AskMyDocs", page_icon="📄", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, button, input, textarea {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* App background */
.stApp {
    background: #07070f;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #0c0c18 !important;
    border-right: 1px solid rgba(124, 58, 237, 0.12) !important;
}
[data-testid="stSidebarContent"] {
    background: transparent !important;
    padding: 1.75rem 1.25rem !important;
}

/* Sidebar logo */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(124, 58, 237, 0.1);
    margin-bottom: 0.25rem;
}
.sidebar-logo-icon {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #6d28d9, #a78bfa);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    box-shadow: 0 4px 16px rgba(109, 40, 217, 0.4);
    flex-shrink: 0;
}
.sidebar-logo-text {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e4e4f0;
    letter-spacing: -0.02em;
}
.sidebar-logo-badge {
    font-size: 0.58rem;
    font-weight: 600;
    color: #a78bfa;
    background: rgba(124, 58, 237, 0.15);
    border: 1px solid rgba(124, 58, 237, 0.25);
    padding: 1px 5px;
    border-radius: 4px;
    margin-left: 5px;
    vertical-align: middle;
    letter-spacing: 0;
}

/* Section labels */
.section-label {
    font-size: 0.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    color: #3d3d58 !important;
    text-transform: uppercase !important;
    margin: 1.4rem 0 0.6rem !important;
    padding: 0 !important;
}

/* Document card */
.doc-card {
    background: rgba(124, 58, 237, 0.06);
    border: 1px solid rgba(124, 58, 237, 0.16);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-top: 0.4rem;
}
.doc-card-name {
    font-size: 0.8rem;
    font-weight: 600;
    color: #cccce0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 0.45rem;
}
.doc-card-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.72rem;
    color: #5a5a78;
}
.doc-badge {
    font-size: 0.66rem;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 20px;
    white-space: nowrap;
}
.doc-badge-text {
    background: rgba(16, 185, 129, 0.09);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.18);
}
.doc-badge-scanned {
    background: rgba(245, 158, 11, 0.09);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.18);
}
.doc-scanned-note {
    font-size: 0.7rem;
    color: #92400e;
    margin-top: 0.5rem;
    line-height: 1.4;
}

/* Capability pills */
.caps-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 0.1rem;
}
.cap-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 20px;
    padding: 3px 9px;
    font-size: 0.68rem;
    color: #44445e;
}

/* Sidebar footer */
.sidebar-footer {
    margin-top: 3rem;
    padding-top: 1.25rem;
    border-top: 1px solid rgba(124, 58, 237, 0.08);
    font-size: 0.7rem;
    color: #2e2e46;
    line-height: 1.7;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(124, 58, 237, 0.04) !important;
    border: 1.5px dashed rgba(124, 58, 237, 0.28) !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(167, 139, 250, 0.5) !important;
    background: rgba(124, 58, 237, 0.07) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #3a3a56 !important;
    font-size: 0.78rem !important;
}
[data-testid="stFileUploader"] svg { opacity: 0.35; }

/* ── MAIN AREA ── */
.main-header {
    text-align: center;
    padding: 2.75rem 1rem 1.75rem;
}
.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.045em;
    background: linear-gradient(135deg, #c4b5fd 0%, #8b5cf6 45%, #6d28d9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.55rem;
}
.main-subtitle {
    font-size: 0.9rem;
    color: #2e2e48;
    font-weight: 400;
    line-height: 1.5;
    max-width: 480px;
    margin: 0 auto;
}
.main-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124,58,237,0.15), transparent);
    margin: 1.5rem 0 0;
}

/* Empty state */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 5rem 2rem;
    text-align: center;
}
.empty-icon {
    font-size: 2.25rem;
    margin-bottom: 1.25rem;
    opacity: 0.15;
}
.empty-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #2a2a42;
    margin-bottom: 0.35rem;
}
.empty-sub {
    font-size: 0.8rem;
    color: #1e1e34;
    max-width: 240px;
    line-height: 1.55;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.35rem 0 !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: rgba(255, 255, 255, 0.025) !important;
    border: 1px solid rgba(124, 58, 237, 0.22) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(124, 58, 237, 0.45) !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.08) !important;
}
[data-testid="stChatInput"] textarea {
    color: #c8c8e0 !important;
    background: transparent !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #2c2c48 !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.81rem !important;
}

/* Spinner */
[data-testid="stSpinner"] > div {
    border-top-color: #7c3aed !important;
}

/* Global text */
.stApp p, .stMarkdown p, .stApp li, .stMarkdown li {
    color: #a0a0c0 !important;
}
.stApp h1, .stApp h2, .stApp h3 {
    color: #d0d0e8 !important;
}
.stApp label {
    color: #5a5a78 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124, 58, 237, 0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124, 58, 237, 0.4); }
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE ─────────────────────────────────────────────────────────
if "index" not in st.session_state:
    st.session_state.index = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "page_images" not in st.session_state:
    st.session_state.page_images = {}
if "is_scanned" not in st.session_state:
    st.session_state.is_scanned = False


# ── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">📄</div>
        <div class="sidebar-logo-text">
            AskMyDocs<span class="sidebar-logo-badge">v2</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-label">Document</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Supports text-based and scanned PDFs.",
    )

    if uploaded_file and uploaded_file.name != st.session_state.doc_name:
        with st.spinner("Indexing document..."):
            chunks = load_and_chunk_pdf(uploaded_file)
            is_scanned = len(chunks) == 0
            st.session_state.is_scanned = is_scanned
            if is_scanned:
                chunks = [
                    "[Scanned document — no extractable text. "
                    "Use analyze_figure to read specific pages visually.]"
                ]
            st.session_state.index = build_index(chunks)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            st.session_state.page_images = render_pdf_pages(tmp_path)
            st.session_state.doc_name = uploaded_file.name
            st.session_state.messages = []

    if st.session_state.doc_name:
        page_count = len(st.session_state.page_images)
        if st.session_state.is_scanned:
            badge = '<span class="doc-badge doc-badge-scanned">Scanned PDF</span>'
            note = '<div class="doc-scanned-note">Visual analysis mode — text search unavailable</div>'
        else:
            badge = '<span class="doc-badge doc-badge-text">Text PDF</span>'
            note = ""
        st.markdown(f"""
        <div class="doc-card">
            <div class="doc-card-name">📄 {st.session_state.doc_name}</div>
            <div class="doc-card-meta">{badge}&nbsp;&nbsp;{page_count} pages</div>
            {note}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="section-label">Capabilities</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="caps-grid">
        <span class="cap-pill">🔍 Semantic search</span>
        <span class="cap-pill">🧮 Calculator</span>
        <span class="cap-pill">🌐 Web search</span>
        <span class="cap-pill">👁️ Vision</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-footer">
        Built by Rucha Joshi<br>
        <span style="opacity:0.45;">Powered by Claude Sonnet 4.6</span>
    </div>
    """, unsafe_allow_html=True)


# ── MAIN AREA ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-title">AskMyDocs</div>
    <div class="main-subtitle">
        Ask anything about your document. Claude searches, calculates, and reasons to find the answer.
    </div>
    <div class="main-divider"></div>
</div>
""", unsafe_allow_html=True)

if st.session_state.index:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"].replace("$", r"\$"))

    question = st.chat_input("Ask a question about your document...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = run_agent(
                        question,
                        st.session_state.index,
                        st.session_state.page_images,
                        is_scanned=st.session_state.is_scanned,
                    )
                    st.markdown(answer.replace("$", r"\$"))
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📋</div>
        <div class="empty-title">No document loaded</div>
        <div class="empty-sub">Upload a PDF from the sidebar to start asking questions about it</div>
    </div>
    """, unsafe_allow_html=True)
