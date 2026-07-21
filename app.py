import streamlit as st
import os
import datetime
import tempfile
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

# ── API KEY ───────────────────────────────────────────────────────────────
# Resolve the key BEFORE importing the pipeline (askmydocs_core reads it at
# import time). Check .env first, then Streamlit secrets (for Streamlit Cloud).
load_dotenv(Path(__file__).with_name(".env"))
if not os.getenv("ANTHROPIC_API_KEY"):
    try:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass

if not os.getenv("ANTHROPIC_API_KEY"):
    st.set_page_config(page_title="AskMyDocs", page_icon="📄", layout="centered")
    st.error(
        "No Anthropic API key found. Add a `.env` file next to app.py containing "
        "`ANTHROPIC_API_KEY=sk-ant-...`, or set it in Streamlit secrets, then restart the app."
    )
    st.stop()

# ── PIPELINE ──────────────────────────────────────────────────────────────
# The retrieval + agent pipeline lives in askmydocs_core so the eval harness
# (testset.py) can import and exercise the exact same code.
from askmydocs_core import (
    load_and_chunk_pdf,
    build_index,
    render_pdf_pages,
    run_agent,
)


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

/* Upload card — top half of the unified upload panel */
.upload-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-bottom: none;
    border-radius: 16px 16px 0 0;
    padding: 2.5rem 2rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    text-align: center;
    margin-bottom: 0 !important;
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
    margin-bottom: 0;
}

/* File uploader — bottom half of the unified upload panel */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-top: none !important;
    border-radius: 0 0 16px 16px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05) !important;
    margin-top: 0 !important;
    padding: 0.25rem 1.5rem 1.5rem !important;
}
/* Restore normal card look for the "change document" expander uploader */
[data-testid="stExpander"] [data-testid="stFileUploader"] {
    border: 1.5px dashed #d1d5db !important;
    border-top: 1.5px dashed #d1d5db !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    padding: 0.5rem !important;
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
    font-size: 0.82rem;
    color: #374151;
    font-weight: 500;
    flex-wrap: wrap;
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

/* Upload hint text */
.upload-hint {
    font-size: 0.82rem;
    color: #6b7280;
    margin: 0.6rem 0 0;
    line-height: 1.5;
}

/* Reset button */
[data-testid="stBaseButton-secondary"] {
    background: #fff1f2 !important;
    border: 1px solid #fecdd3 !important;
    color: #be123c !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}
[data-testid="stBaseButton-secondary"]:hover {
    background: #ffe4e6 !important;
    border-color: #fda4af !important;
}

/* Inline download icon button — sits in a column beside the chat input */
[data-testid="stDownloadButton"] button {
    width: 45px;
    height: 45px;
    min-height: 45px;
    padding: 0 !important;
    border-radius: 50% !important;
    background: #ffffff !important;
    border: 1.5px solid #e5e7eb !important;
    color: #1d4ed8 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: #1d4ed8 !important;
    box-shadow: 0 3px 12px rgba(0,0,0,0.12) !important;
    transform: translateY(-7px);
}
[data-testid="stDownloadButton"] button p { font-size: 1.6rem !important; }
/* Raise the download button so it's flush with the send arrow in the chat box */
[data-testid="stDownloadButton"] { margin-bottom: 6px; }
[data-testid="stDownloadButton"] button { transform: translateY(3px); }

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
if "documents" not in st.session_state:
    st.session_state.documents = []   # [{name, is_scanned, page_start, page_count}]
if "page_images" not in st.session_state:
    st.session_state.page_images = {}


def reset_all():
    st.session_state.index = None
    st.session_state.documents = []
    st.session_state.page_images = {}
    st.session_state.messages = []


def process_upload(uploaded_file) -> bool:
    if not uploaded_file:
        return False
    if uploaded_file.name in {d["name"] for d in st.session_state.documents}:
        return False
    with st.spinner(f"Indexing {uploaded_file.name}..."):
        chunks = load_and_chunk_pdf(uploaded_file)
        is_scanned = len(chunks) == 0
        if is_scanned:
            chunks = ["[Scanned document — no extractable text. Use analyze_figure for page-specific questions.]"]
        # Prefix chunks with source name so multi-doc search is attributable
        prefixed = [f"[Source: {uploaded_file.name}] {c}" for c in chunks]
        new_index = build_index(prefixed)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        new_pages = render_pdf_pages(tmp_path)
        offset = len(st.session_state.page_images)
        st.session_state.page_images.update({k + offset: v for k, v in new_pages.items()})
        if st.session_state.index is None:
            st.session_state.index = new_index
        else:
            st.session_state.index = {
                "chunks": st.session_state.index["chunks"] + new_index["chunks"],
                "embeddings": np.concatenate(
                    [st.session_state.index["embeddings"], new_index["embeddings"]], axis=0
                ),
            }
        st.session_state.documents.append({
            "name": uploaded_file.name,
            "is_scanned": is_scanned,
            "page_start": offset + 1,
            "page_count": len(new_pages),
        })
    return True


def format_conversation_md() -> str:
    doc_names = ", ".join(d["name"] for d in st.session_state.documents)
    lines = [
        "# AskMyDocs Conversation\n",
        f"**Document(s):** {doc_names}  ",
        f"**Date:** {datetime.date.today().isoformat()}\n",
        "\n---\n",
    ]
    for msg in st.session_state.messages:
        role = "You" if msg["role"] == "user" else "AskMyDocs"
        lines.append(f"\n**{role}:** {msg['content']}\n")
    return "\n".join(lines)


# ── SIDEBAR — doc info panel (desktop bonus; all critical UI is in main area) ──
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">📄</div>
        <div class="sidebar-logo-text">AskMyDocs<span class="sidebar-logo-badge">v2</span></div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.documents:
        st.markdown('<p class="section-label">Documents</p>', unsafe_allow_html=True)
        for doc in st.session_state.documents:
            badge = (
                '<span class="doc-badge doc-badge-scanned">Scanned</span>'
                if doc["is_scanned"]
                else '<span class="doc-badge doc-badge-text">Text PDF</span>'
            )
            st.markdown(f"""
            <div class="doc-card" style="margin-bottom:0.5rem;">
                <div class="doc-card-name">📄 {doc["name"]}</div>
                <div class="doc-card-meta">{badge}&nbsp;&nbsp;{doc["page_count"]} pages</div>
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

has_docs = len(st.session_state.documents) > 0

# ── UPLOAD PANEL (always visible — add first doc or add more) ─────────────
if has_docs:
    card_title = "Add another document"
    card_sub = "Upload a second PDF to ask questions across both at once"
    caps_html = ""
else:
    card_title = "Upload a document to get started"
    card_sub = "Supports text-based and scanned PDFs · Up to 200 MB"
    caps_html = """<div class="caps-row">
        <span class="cap-pill">🔍 Semantic search</span>
        <span class="cap-pill">🧮 Calculator</span>
        <span class="cap-pill">🌐 Web search</span>
        <span class="cap-pill">👁️ Vision</span>
    </div>"""

st.markdown(f"""
<div class="upload-card">
    <div class="upload-card-icon">📄</div>
    <div class="upload-card-title">{card_title}</div>
    <div class="upload-card-sub">{card_sub}</div>
    {caps_html}
</div>
""", unsafe_allow_html=True)

# Key changes on every new doc so the widget resets cleanly
uploader_key = f"uploader_{len(st.session_state.documents)}"
uploaded_file = st.file_uploader(
    "PDF", type=["pdf"], key=uploader_key, label_visibility="collapsed",
    help="Drag and drop or click Browse. Supports text-based and scanned PDFs.",
)
if process_upload(uploaded_file):
    st.rerun()

# ── HINT + RESET (shown after first upload) ───────────────────────────────
if has_docs:
    if len(st.session_state.documents) == 1:
        doc = st.session_state.documents[0]
        hint = f"✓ <strong>{doc['name']}</strong> ready. Upload another PDF above to ask questions across both, or start chatting below."
    else:
        names = " &amp; ".join(f"<strong>{d['name']}</strong>" for d in st.session_state.documents)
        hint = f"✓ {names} loaded. You can add more documents above or start chatting below."

    col_hint, col_reset = st.columns([5, 1])
    with col_hint:
        st.markdown(f'<p class="upload-hint">{hint}</p>', unsafe_allow_html=True)
    with col_reset:
        if st.button("Reset", use_container_width=True):
            reset_all()
            st.rerun()

    st.markdown('<div class="main-divider" style="margin:1.25rem 0;"></div>', unsafe_allow_html=True)

    # ── CHAT ─────────────────────────────────────────────────────────────

    # Status bar + download button on same row
    doc_badges = " ".join(
        f'<span class="doc-badge {"doc-badge-scanned" if d["is_scanned"] else "doc-badge-text"}">'
        f'{d["name"]}</span>'
        for d in st.session_state.documents
    )
    total_pages = len(st.session_state.page_images)
    st.markdown(f"""
    <div class="doc-status-bar">
        {doc_badges}&nbsp;&nbsp;
        <span style="color:#9ca3af;">{total_pages} page{"s" if total_pages != 1 else ""} total</span>
    </div>
    """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("trace"):
                with st.expander("🔎 Show reasoning steps"):
                    for step in msg["trace"]:
                        st.markdown(f"- {step}".replace("$", r"\$"))
            st.markdown(msg["content"].replace("$", r"\$"))

    # Chat input and the download icon share one row so they stay aligned
    has_reply = any(m["role"] == "assistant" for m in st.session_state.messages)
    col_input, col_dl = st.columns([9, 1], vertical_alignment="bottom")
    with col_input:
        question = st.chat_input("Ask a question about your document(s)...")
    with col_dl:
        if has_reply:
            st.download_button(
                "⬇",
                data=format_conversation_md(),
                file_name=f"askmydocs_{datetime.date.today().isoformat()}.md",
                mime="text/markdown",
                help="Download conversation",
                key="download_conv",
            )

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, trace = run_agent(
                        question,
                        st.session_state.index,
                        st.session_state.page_images,
                        documents=st.session_state.documents,
                    )
                    if trace:
                        with st.expander("🔎 Show reasoning steps"):
                            for step in trace:
                                st.markdown(f"- {step}".replace("$", r"\$"))
                    st.markdown(answer.replace("$", r"\$"))
                    st.session_state.messages.append({
                        "role": "assistant", "content": answer, "trace": trace,
                    })
                except Exception as e:
                    st.error(f"Error: {e}")
