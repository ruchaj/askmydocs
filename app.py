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
    if page_number not in page_images:
        valid = sorted(page_images.keys())
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


def execute_tool(name: str, tool_input: dict, index: dict, page_images: dict[int, str]) -> str:
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
    page_count = len(page_images)
    system = SYSTEM_PROMPT + f" The document has {page_count} page(s); valid page numbers for analyze_figure are 1 to {page_count}."
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
st.set_page_config(page_title="AskMyDocs", page_icon="📄", layout="centered")

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, button, input, textarea {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* App background — light warm gray */
.stApp {
    background: #f4f6f8;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #dde1e7 !important;
    box-shadow: 2px 0 8px rgba(0, 0, 0, 0.04) !important;
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
    border-bottom: 1px solid #eaedf0;
    margin-bottom: 0.25rem;
}
.sidebar-logo-icon {
    width: 38px;
    height: 38px;
    background: #1d4ed8;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    box-shadow: 0 2px 8px rgba(29, 78, 216, 0.3);
    flex-shrink: 0;
}
.sidebar-logo-text {
    font-size: 1.05rem;
    font-weight: 700;
    color: #111827;
    letter-spacing: -0.02em;
}
.sidebar-logo-badge {
    font-size: 0.58rem;
    font-weight: 600;
    color: #1d4ed8;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    padding: 1px 5px;
    border-radius: 4px;
    margin-left: 5px;
    vertical-align: middle;
}

/* Section labels */
.section-label {
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    color: #9ca3af !important;
    text-transform: uppercase !important;
    margin: 1.4rem 0 0.6rem !important;
    padding: 0 !important;
}

/* Document card */
.doc-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-top: 0.4rem;
}
.doc-card-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: #111827;
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
    color: #6b7280;
}
.doc-badge {
    font-size: 0.66rem;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 20px;
    white-space: nowrap;
}
.doc-badge-text {
    background: #dcfce7;
    color: #166534;
    border: 1px solid #bbf7d0;
}
.doc-badge-scanned {
    background: #fef9c3;
    color: #854d0e;
    border: 1px solid #fde047;
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
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.69rem;
    color: #6b7280;
}

/* Sidebar footer */
.sidebar-footer {
    margin-top: 3rem;
    padding-top: 1.25rem;
    border-top: 1px solid #eaedf0;
    font-size: 0.7rem;
    color: #9ca3af;
    line-height: 1.7;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #f9fafb !important;
    border: 1.5px dashed #d1d5db !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #1d4ed8 !important;
    background: #eff6ff !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #9ca3af !important;
    font-size: 0.78rem !important;
}

/* ── MAIN AREA ── */
.main-header {
    text-align: center;
    padding: 2.75rem 1rem 1.75rem;
}
.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.045em;
    color: #111827;
    line-height: 1.1;
    margin-bottom: 0.55rem;
}
.main-title .accent {
    color: #1d4ed8;
}
.main-subtitle {
    font-size: 0.95rem;
    color: #6b7280;
    font-weight: 400;
    line-height: 1.6;
    max-width: 480px;
    margin: 0 auto;
}
.main-divider {
    height: 1px;
    background: #e5e7eb;
    margin: 1.75rem 0 0;
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
    font-size: 2.5rem;
    margin-bottom: 1.25rem;
    opacity: 0.3;
}
.empty-title {
    font-size: 1rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 0.4rem;
}
.empty-sub {
    font-size: 0.85rem;
    color: #9ca3af;
    max-width: 260px;
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
    background: #ffffff !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #1d4ed8 !important;
    box-shadow: 0 0 0 3px rgba(29, 78, 216, 0.08) !important;
}
[data-testid="stChatInput"] textarea {
    color: #111827 !important;
    background: transparent !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #9ca3af !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.82rem !important;
}

/* Spinner */
[data-testid="stSpinner"] > div {
    border-top-color: #1d4ed8 !important;
}

/* Global text overrides */
.stApp p, .stMarkdown p, .stApp li, .stMarkdown li {
    color: #374151 !important;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: #111827 !important;
}
.stApp label, .stApp span {
    color: #6b7280 !important;
}
strong, b { color: #111827 !important; }

/* Upload card */
.upload-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 2.5rem 2rem 1.75rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04), 0 6px 24px rgba(0,0,0,0.04);
    text-align: center;
    margin-bottom: 0.75rem;
}
.upload-card-icon { font-size: 2.25rem; margin-bottom: 1rem; }
.upload-card-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.35rem;
}
.upload-card-sub {
    font-size: 0.85rem;
    color: #6b7280;
    margin-bottom: 1.25rem;
    line-height: 1.5;
}
.caps-row {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 1.25rem;
}

/* Doc status bar (shown above chat on all devices) */
.doc-status-bar {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.82rem;
    color: #374151;
    font-weight: 500;
}

/* Expander (change document) */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    margin-bottom: 1rem !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.85rem !important;
    color: #374151 !important;
    font-weight: 500 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f4f6f8; }
::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #9ca3af; }
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


def process_upload(uploaded_file) -> bool:
    """Index an uploaded PDF and update session state. Returns True if a new file was loaded."""
    if not uploaded_file or uploaded_file.name == st.session_state.doc_name:
        return False
    with st.spinner("Reading and indexing your document..."):
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
    return True


# ── SIDEBAR — doc info only (desktop bonus; not required on mobile) ────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">📄</div>
        <div class="sidebar-logo-text">
            AskMyDocs<span class="sidebar-logo-badge">v2</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.doc_name:
        page_count = len(st.session_state.page_images)
        badge = (
            '<span class="doc-badge doc-badge-scanned">Scanned PDF</span>'
            if st.session_state.is_scanned
            else '<span class="doc-badge doc-badge-text">Text PDF</span>'
        )
        note = (
            '<div class="doc-scanned-note">Visual analysis mode — text search unavailable</div>'
            if st.session_state.is_scanned else ""
        )
        st.markdown(f"""
        <p class="section-label">Document</p>
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
        <span style="opacity:0.6;">Powered by Claude Sonnet 4.6</span>
    </div>
    """, unsafe_allow_html=True)


# ── MAIN AREA ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-title">Ask<span class="accent">My</span>Docs</div>
    <div class="main-subtitle">
        Ask anything about your document. Claude searches, calculates, and reasons to find the answer.
    </div>
    <div class="main-divider"></div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.index:
    # ── UPLOAD STATE: prominent upload card, works on all screen sizes ────
    st.markdown("""
    <div class="upload-card">
        <div class="upload-card-icon">📄</div>
        <div class="upload-card-title">Upload a document to get started</div>
        <div class="upload-card-sub">Supports text-based and scanned PDFs · Up to 200 MB</div>
        <div class="caps-row">
            <span class="cap-pill">🔍 Semantic search</span>
            <span class="cap-pill">🧮 Calculator</span>
            <span class="cap-pill">🌐 Web search</span>
            <span class="cap-pill">👁️ Vision</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "PDF",
        type=["pdf"],
        key="uploader",
        label_visibility="collapsed",
        help="Drag and drop or click Browse. Supports text-based and scanned PDFs.",
    )
    if process_upload(uploaded_file):
        st.rerun()

else:
    # ── CHAT STATE ────────────────────────────────────────────────────────

    # Doc status bar — visible on all devices (mobile has no sidebar)
    page_count = len(st.session_state.page_images)
    badge_html = (
        '<span class="doc-badge doc-badge-scanned">Scanned</span>'
        if st.session_state.is_scanned
        else '<span class="doc-badge doc-badge-text">Text PDF</span>'
    )
    st.markdown(f"""
    <div class="doc-status-bar">
        📄 {st.session_state.doc_name}&nbsp;&nbsp;{badge_html}&nbsp;&nbsp;
        <span style="color:#9ca3af;">{page_count} pages</span>
    </div>
    """, unsafe_allow_html=True)

    # Change document — collapsed by default, always accessible on mobile
    with st.expander("Upload a different document"):
        new_file = st.file_uploader(
            "PDF",
            type=["pdf"],
            key="uploader_change",
            label_visibility="collapsed",
        )
        if process_upload(new_file):
            st.rerun()

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"].replace("$", r"\$"))

    # Chat input
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
