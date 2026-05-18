# ── IMPORTS ──────────────────────────────────────────────────────────────
# streamlit turns this Python script into a web app
import streamlit as st
# anthropic is the official SDK to talk to Claude API
import anthropic
# chromadb is our vector database — stores and searches embeddings
import chromadb
# json handles converting between Python dicts and JSON text
import json
# os lets us read environment variables (our API key)
import os
# pypdf reads and extracts text from PDF files
from pypdf import PdfReader
# sentence_transformers converts text into numerical vectors (embeddings)
from sentence_transformers import SentenceTransformer
# dotenv reads our .env file so our API key stays secret
from dotenv import load_dotenv

# ── SETUP ─────────────────────────────────────────────────────────────────
# load the .env file so os.getenv() can find our API key
load_dotenv()

# create the Claude client — our connection to the Claude API
# os.getenv() reads ANTHROPIC_API_KEY from the .env file
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# load the embedding model — this converts text chunks into vectors
# all-MiniLM-L6-v2 is small, fast, and free — perfect for local use
# this line runs once when the app starts and is cached by Streamlit
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# create an in-memory ChromaDB client
# in-memory means it resets every time the app restarts — fine for demos
chroma = chromadb.Client()


# ── FUNCTION 1: LOAD AND CHUNK PDF ───────────────────────────────────────
def load_and_chunk_pdf(uploaded_file, chunk_size=500):
    # PdfReader opens the PDF and lets us access each page
    reader = PdfReader(uploaded_file)

    # extract text from every page and join into one big string
    # the 'if page.extract_text()' skips blank or image-only pages
    full_text = " ".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )

    # split the full text into individual words
    words = full_text.split()

    # create overlapping chunks of ~500 words
    # overlap of 50 words (chunk_size - 50) prevents losing context at boundaries
    # e.g. if a sentence spans two chunks, the overlap captures it in both
    chunks = []
    for i in range(0, len(words), chunk_size - 50):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:  # skip empty chunks
            chunks.append(chunk)

    return chunks  # returns a list of text strings


# ── FUNCTION 2: BUILD VECTOR INDEX ───────────────────────────────────────
def build_index(chunks):
    col_name = "documents"

    # delete existing collection if it exists (handles re-uploading a new PDF)
    try:
        chroma.delete_collection(col_name)
    except:
        pass  # if it doesn't exist yet, that's fine — ignore the error

    # create a fresh ChromaDB collection (like a table in a database)
    collection = chroma.create_collection(col_name)

    # convert all text chunks into embeddings (numerical vectors)
    # embedder.encode() returns a numpy array — .tolist() converts to plain Python list
    # this is the most computationally expensive step — takes a few seconds
    embeddings = embedder.encode(chunks).tolist()

    # store chunks AND their embeddings in ChromaDB
    # documents = the original text (so we can return it to the user)
    # embeddings = the vectors (so we can do similarity search)
    # ids = unique identifier for each chunk (required by ChromaDB)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    return collection  # return the collection so we can query it later


# ── FUNCTION 3: ANSWER A QUESTION ────────────────────────────────────────
def answer_question(question, collection):
    # convert the user's question into an embedding vector
    # same model as before — so the vector space is compatible
    q_embedding = embedder.encode([question]).tolist()

    # search ChromaDB for the 3 most similar chunks to the question
    # this is semantic search — finds meaning, not just keyword matches
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=3  # retrieve top 3 most relevant chunks
    )

    # join the 3 retrieved chunks into one context string
    # results['documents'][0] is a list of the matching text chunks
    context = "\n\n".join(results['documents'][0])

    # send the context + question to Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,

        # system prompt tells Claude its role and constraints
        # "Answer only using the context" prevents hallucination
        system="""You are a helpful assistant that answers questions about documents.
Answer only using the context provided. If the answer is not in the context,
say 'I could not find that information in the document.'
Always be specific and quote relevant parts when helpful.""",

        # messages is the conversation — here just one user turn
        # we inject the retrieved context + the user's question
        messages=[{
            "role": "user",
            "content": f"Context from document:\n{context}\n\nQuestion: {question}"
        }]
    )

    # return Claude's text answer AND the source chunks (to show the user)
    return response.content[0].text, results['documents'][0]


# ── STREAMLIT UI ──────────────────────────────────────────────────────────

# configure the browser tab title, icon, and layout
st.set_page_config(page_title="AskMyDocs", page_icon="📄", layout="wide")

# render the app header
st.title("AskMyDocs")
st.caption("Upload a PDF · Ask questions in plain English · Powered by Claude · Built by Rucha Joshi")
# custom CSS styling
st.markdown("""
<style>
    /* main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }

    /* all text white */
    .stApp, .stApp p, .stApp label, .stMarkdown {
        color: white !important;
    }

    /* title styling */
    h1 { color: #a78bfa !important; font-size: 3rem !important; }
    h2, h3 { color: #7c3aed !important; }

    /* file uploader box */
    .stFileUploader {
        background: rgba(255,255,255,0.05);
        border: 2px dashed #7c3aed;
        border-radius: 12px;
        padding: 1rem;
    }

    /* chat input box */
    .stChatInput input {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid #7c3aed !important;
        color: white !important;
        border-radius: 20px !important;
    }

    /* user chat bubble */
    .stChatMessage[data-testid="chat-message-user"] {
        background: rgba(124, 58, 237, 0.3);
        border-radius: 12px;
        padding: 0.5rem;
    }

    /* assistant chat bubble */
    .stChatMessage[data-testid="chat-message-assistant"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 0.5rem;
    }

    /* success message */
    .stSuccess {
        background: rgba(16, 185, 129, 0.2) !important;
        border: 1px solid #10b981 !important;
        border-radius: 8px !important;
        color: #10b981 !important;
    }

    /* spinner */
    .stSpinner { color: #a78bfa !important; }

    /* expander */
    .streamlit-expanderHeader {
        background: rgba(124, 58, 237, 0.2) !important;
        border-radius: 8px !important;
        color: white !important;
    }

    /* divider */
    hr { border-color: #7c3aed !important; opacity: 0.3; }

    /* caption text */
    .stCaption { color: #a78bfa !important; }
</style>
""", unsafe_allow_html=True)
st.divider()

# ── SESSION STATE ─────────────────────────────────────────────────────────
# Streamlit re-runs the entire script on every user interaction
# st.session_state persists variables across re-runs — like a memory store
# without this, the collection and chat history would reset on every click

# collection = the ChromaDB index of the current document
if "collection" not in st.session_state:
    st.session_state.collection = None

# messages = the full chat history (list of {role, content} dicts)
if "messages" not in st.session_state:
    st.session_state.messages = []

# doc_name = tracks which PDF is currently loaded
# used to detect when the user uploads a NEW document
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ── FILE UPLOAD ───────────────────────────────────────────────────────────
# render the file uploader widget — only accepts PDF files
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# only re-index if a NEW file is uploaded (different name than current)
# this prevents re-indexing the same document on every interaction
if uploaded_file and uploaded_file.name != st.session_state.doc_name:
    with st.spinner("Reading and indexing your document..."):
        # step 1: extract text and split into chunks
        chunks = load_and_chunk_pdf(uploaded_file)
        # step 2: embed chunks and store in ChromaDB
        st.session_state.collection = build_index(chunks)
        # step 3: save the filename so we don't re-index unnecessarily
        st.session_state.doc_name = uploaded_file.name
        # step 4: clear chat history when a new document is loaded
        st.session_state.messages = []

    # show a success message with how many sections were indexed
    st.success(f"Ready! Indexed {len(chunks)} sections from {uploaded_file.name}")

# ── CHAT INTERFACE ────────────────────────────────────────────────────────
# only show the chat if a document has been indexed
if st.session_state.collection:
    st.subheader("Chat with your document")

    # render the full chat history
    # loops through all previous messages and displays them in order
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):  # "user" or "assistant"
            st.write(msg["content"])

    # render the chat input box at the bottom of the page
    # st.chat_input() returns the user's message when they press Enter
    question = st.chat_input("Ask a question about your document...")

    if question:
        # add the user's question to chat history
        st.session_state.messages.append({"role": "user", "content": question})

        # display the user's message immediately
        with st.chat_message("user"):
            st.write(question)

        # generate and display Claude's answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # call our answer_question function
                    # returns Claude's answer + the source chunks used
                    answer, sources = answer_question(
                        question, st.session_state.collection
                    )

                    # display Claude's answer
                    st.write(answer)

                    # show the source sections in a collapsible expander
                    # this shows the user WHERE the answer came from
                    with st.expander("Source sections used"):
                        for i, src in enumerate(sources, 1):
                            st.caption(f"Section {i}: {src[:300]}...")

                    # save Claude's answer to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    # if anything goes wrong, show a friendly error
                    # e contains the actual error message for debugging
                    st.error(f"Error: {e}")

else:
    # show a hint if no document is uploaded yet
    st.info("Upload a PDF above to get started")