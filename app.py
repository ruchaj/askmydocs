# ── IMPORTS ──────────────────────────────────────────────────────────────
# streamlit turns this Python script into a web app — no HTML needed
import streamlit as st
# anthropic is the official SDK to talk to Claude API
import anthropic
# os lets us read environment variables like our API key
import os
# numpy is a math library — we use it for vector similarity calculations
import numpy as np
# pypdf reads and extracts text from PDF files
from pypdf import PdfReader
# sentence_transformers converts text into numerical vectors (embeddings)
# all-MiniLM-L6-v2 is a small, fast, free model — perfect for this use case
from sentence_transformers import SentenceTransformer
# dotenv reads our .env file so our API key stays secret
from dotenv import load_dotenv

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


# ── FUNCTION 1: LOAD AND CHUNK PDF ───────────────────────────────────────
def load_and_chunk_pdf(uploaded_file, chunk_size=500):
    # PdfReader opens the uploaded PDF file
    reader = PdfReader(uploaded_file)

    # loop through every page, extract text, skip blank/image-only pages
    # join all pages into one long string with spaces between them
    full_text = " ".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )

    # split the full text into individual words
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


# ── FUNCTION 3: ANSWER A QUESTION ────────────────────────────────────────
def answer_question(question, index):
    # convert the user's question into an embedding vector
    # [0] gets the first (only) result since we passed a list of one item
    q_embedding = embedder.encode([question])[0]

    # retrieve stored embeddings and chunks from our index
    embeddings = index["embeddings"]
    chunks = index["chunks"]

    # calculate cosine similarity between the question and every chunk
    # cosine similarity measures the ANGLE between two vectors
    # closer to 1.0 = more similar meaning, closer to 0 = unrelated
    # formula: dot_product / (magnitude_A * magnitude_B)
    # the + 1e-10 prevents division by zero errors
    similarities = np.dot(embeddings, q_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_embedding) + 1e-10
    )

    # np.argsort() returns indices sorted from lowest to highest similarity
    # [-3:] takes the last 3 (the highest similarity scores)
    # [::-1] reverses to get highest first
    # result: indices of the 3 most relevant chunks
    top_3 = np.argsort(similarities)[-3:][::-1]

    # get the actual text of the top 3 most relevant chunks
    top_chunks = [chunks[i] for i in top_3]

    # join the 3 chunks into one context string to send to Claude
    context = "\n\n".join(top_chunks)

    # send the retrieved context + question to Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,

        # system prompt defines Claude's role and constraints
        # "Answer only using the context" is critical — it prevents hallucination
        # without this constraint, Claude might make up answers not in the document
        system="""You are a helpful assistant that answers questions about documents.
Answer only using the context provided. If the answer is not in the context,
say 'I could not find that information in the document.'
Always be specific and quote relevant parts when helpful.""",

        # inject the retrieved context and question into the user message
        # this is the RAG pattern: Retrieve → Augment → Generate
        messages=[{
            "role": "user",
            "content": f"Context from document:\n{context}\n\nQuestion: {question}"
        }]
    )

    # return Claude's answer text AND the source chunks
    # source chunks are shown to the user in the "Source sections used" expander
    return response.content[0].text, top_chunks


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


# ── STREAMLIT PAGE CONFIG ─────────────────────────────────────────────────
# must be the first Streamlit command — sets browser tab title and layout
st.set_page_config(page_title="AskMyDocs", page_icon="📄", layout="wide")

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


# ── FILE UPLOAD ───────────────────────────────────────────────────────────
# renders the file upload widget — restricts to PDF only
# returns None if no file uploaded, or a file object if one is selected
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# only process if a file is uploaded AND it's different from current document
# uploaded_file.name != st.session_state.doc_name prevents re-processing
# the same file every time the user interacts with the app
if uploaded_file and uploaded_file.name != st.session_state.doc_name:
    with st.spinner("Reading and indexing your document..."):
        # step 1: extract text from PDF and split into overlapping chunks
        chunks = load_and_chunk_pdf(uploaded_file)
        # step 2: convert chunks to embeddings and store in our index
        st.session_state.index = build_index(chunks)
        # step 3: remember the filename to avoid re-processing
        st.session_state.doc_name = uploaded_file.name
        # step 4: clear previous chat history when a new doc is loaded
        st.session_state.messages = []

    # show success message with chunk count
    st.success(f"Ready! Indexed {len(chunks)} sections from {uploaded_file.name}")


# ── CHAT INTERFACE ────────────────────────────────────────────────────────
# only show the chat UI after a document has been indexed
if st.session_state.index:
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
                    answer, sources = answer_question(
                        question, st.session_state.index
                    )

                    # display Claude's answer as the assistant's message
                    st.write(answer)

                    # show source sections in a collapsible expander
                    # this builds user trust — they can see WHERE the answer came from
                    with st.expander("Source sections used"):
                        for i, src in enumerate(sources, 1):
                            # show first 300 chars of each source chunk
                            st.caption(f"Section {i}: {src[:300]}...")

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
    # shown when no document is loaded yet — guides the user
    st.info("Upload a PDF above to get started")