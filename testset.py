"""
AskMyDocs — lightweight RAG eval harness
========================================
Runs the real AskMyDocs pipeline (imported from askmydocs_core) over a corpus of
generated test PDFs and measures three things per question:
  1. Retrieval hit-rate (recall@k) — did the right source chunk get fetched?
  2. Faithfulness                  — is the answer supported by the retrieved context (no hallucination)?
  3. Correctness                   — does the answer match the reference answer?

Faithfulness/correctness use Claude as an LLM-judge (free-form text can't be string-matched).

The corpus is the PDFs in test_pdfs/, produced by make_test_pdfs.py (auto-generated
on first run if missing). TEST_SET below is written against that corpus's content.

USAGE
  Run:  python testset.py
  To compare configurations, change ONE variable (CHUNK_SIZE, TOP_K, the prompt),
  re-run, and diff the numbers. Each run is appended to eval_results.jsonl with RUN_LABEL.
  Set DEMO = True to run on a tiny built-in document instead of the PDF corpus.
"""

import json
import re
import time
from pathlib import Path

import numpy as np

# The pipeline is imported from the SAME module the app uses, so this harness
# exercises the real AskMyDocs code (chunking, embeddings, retrieval, agent).
try:
    import askmydocs_core as core
except RuntimeError as e:
    raise SystemExit(str(e))

client = core.client   # reuse the app's Anthropic client for the LLM-judge

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
JUDGE_MODEL = "claude-sonnet-4-6"
TOP_K = 5                       # how many chunks your retriever returns
CHUNK_SIZE = 80                 # word-window size passed to the app's chunker (must be > 50)
RUN_LABEL = "baseline"          # name this run, e.g. "chunk-80", "chunk-120"
DEMO = False                    # True = built-in mini doc; False = read the test_pdfs/ corpus
PDF_DIR = Path(__file__).with_name("test_pdfs")


# ----------------------------------------------------------------------
# TEST SET — questions about the test_pdfs/ corpus (see make_test_pdfs.py).
#   expected_snippet = a distinctive phrase that appears in the correct source
#   chunk, used to score retrieval without needing chunk IDs.
# ----------------------------------------------------------------------
TEST_SET = [
    # ── acme_returns_policy.pdf ──
    {
        "question": "What is the return policy window?",
        "expected_answer": "Items can be returned within 30 days of purchase for a full refund.",
        "expected_snippet": "within 30 days",
    },
    {
        "question": "Does the warranty cover water damage?",
        "expected_answer": "No, water damage is not covered.",
        "expected_snippet": "water damage is not covered",
    },
    {
        "question": "Who qualifies for free shipping?",
        "expected_answer": "Orders over $50 qualify for free standard shipping.",
        "expected_snippet": "free standard shipping",
    },
    # ── helios_financials.pdf ──
    {
        "question": "What was Helios Corp's total revenue in fiscal year 2024?",
        "expected_answer": "$214.6 million.",
        "expected_snippet": "214.6 million",
    },
    {
        "question": "Did Helios take on any new debt in 2024?",
        "expected_answer": "No, the company took on no new debt during the year.",
        "expected_snippet": "took on no new debt",
    },
    {
        "question": "How much did Helios spend on research and development in 2024?",
        "expected_answer": "$32.4 million.",
        "expected_snippet": "32.4 million",
    },
    {
        "question": "How many people did Helios employ at year end?",
        "expected_answer": "1,240 people.",
        "expected_snippet": "1,240 people",
    },
    # ── northwind_handbook.pdf ──
    {
        "question": "How much paid time off do full-time employees get per year?",
        "expected_answer": "15 days of paid time off per year.",
        "expected_snippet": "15 days of paid time off",
    },
    {
        "question": "How many days a week can employees work remotely?",
        "expected_answer": "Up to three days per week with manager approval.",
        "expected_snippet": "three days per week",
    },
    {
        "question": "What is the notice period for resignation?",
        "expected_answer": "Two weeks.",
        "expected_snippet": "two weeks",
    },
    # ── quantum_router_specs.pdf ──
    {
        "question": "What is the maximum throughput of the QR-500 router?",
        "expected_answer": "4.8 Gbps.",
        "expected_snippet": "4.8 gbps",
    },
    {
        "question": "How many Ethernet ports does the QR-500 have?",
        "expected_answer": "Four gigabit Ethernet ports.",
        "expected_snippet": "four gigabit ethernet ports",
    },
    {
        "question": "What is the operating temperature range of the QR-500?",
        "expected_answer": "0 to 40 degrees Celsius.",
        "expected_snippet": "0 to 40 degrees",
    },
    # ── glacier_travel_faq.pdf ──
    {
        "question": "How far in advance can tours be booked?",
        "expected_answer": "Up to six months in advance.",
        "expected_snippet": "six months in advance",
    },
    {
        "question": "What is the cancellation refund policy?",
        "expected_answer": "A full refund minus the deposit if cancelled more than 14 days before departure.",
        "expected_snippet": "14 days before departure",
    },
]


# ----------------------------------------------------------------------
# Pipeline — chunk + embed the corpus once via the app's code, then retrieve
# and answer through it. A parallel `sources` list lets retrieval be scored
# per chunk, and the same index dict is handed to core.run_agent so the agent
# searches it directly.
# ----------------------------------------------------------------------
_INDEX = None       # dict returned by core.build_index: {"chunks":[str], "embeddings": ndarray}
_SOURCES = None     # list[str] aligned with _INDEX["chunks"]: the source PDF per chunk


def _build_index():
    """Chunk + embed the whole PDF corpus once, via the app's pipeline."""
    global _INDEX, _SOURCES
    if _INDEX is not None:
        return

    # Auto-generate the corpus on first run if it isn't there yet.
    if not PDF_DIR.exists() or not list(PDF_DIR.glob("*.pdf")):
        import make_test_pdfs
        make_test_pdfs.generate()

    chunk_texts, sources = [], []
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        with open(pdf_path, "rb") as fh:
            pieces = core.load_and_chunk_pdf(fh, chunk_size=CHUNK_SIZE)
        for piece in pieces:
            chunk_texts.append(piece)
            sources.append(pdf_path.name)

    _INDEX = core.build_index(chunk_texts)   # app's embedder
    _SOURCES = sources


def retrieve(question: str) -> list[dict]:
    """Return the top-k chunks the app's vector search ranks highest for `question`."""
    _build_index()
    q = core.embedder.encode([question])[0]
    embeddings = _INDEX["embeddings"]
    chunks = _INDEX["chunks"]
    sims = np.dot(embeddings, q) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q) + 1e-10
    )
    top = np.argsort(sims)[-TOP_K:][::-1]
    return [{"text": chunks[i], "source": _SOURCES[i]} for i in top]


def answer(question: str) -> tuple[str, list[dict]]:
    """Run the app's full agent pipeline over the corpus. Return (answer_text, chunks_used)."""
    _build_index()
    # The agent retrieves + reasons over the SAME index via its search_documents tool.
    # No page images here (text corpus), so vision is unavailable — text Q&A won't need it.
    answer_text, _trace = core.run_agent(question, _INDEX, page_images={})
    return answer_text, retrieve(question)


# ----------------------------------------------------------------------
# DEMO pipeline — a tiny in-memory "RAG" for a quick smoke test (DEMO = True).
# Naive keyword retrieval, then asks Claude to answer from the retrieved context.
# ----------------------------------------------------------------------
_DEMO_DOC = [
    {"text": "Returns: customers can return items within 30 days of purchase for a full refund.", "source": "policy.pdf p.1"},
    {"text": "Warranty: the limited warranty covers manufacturing defects. Water damage is not covered.", "source": "policy.pdf p.2"},
    {"text": "Shipping: orders over $50 qualify for free standard shipping within the US.", "source": "policy.pdf p.3"},
    {"text": "Support: contact support@example.com for assistance with any order.", "source": "policy.pdf p.4"},
]

def _demo_retrieve(question: str) -> list[dict]:
    qwords = set(re.findall(r"\w+", question.lower()))
    scored = sorted(
        _DEMO_DOC,
        key=lambda c: len(qwords & set(re.findall(r"\w+", c["text"].lower()))),
        reverse=True,
    )
    return scored[:TOP_K]

def _demo_answer(question: str) -> tuple[str, list[dict]]:
    chunks = _demo_retrieve(question)
    context = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks))
    resp = client.messages.create(
        model=JUDGE_MODEL, max_tokens=300,
        system="Answer ONLY from the context. If it isn't there, say you don't know.",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    return resp.content[0].text.strip(), chunks


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def retrieval_hit(chunks: list[dict], expected_snippet: str) -> bool:
    snip = expected_snippet.lower()
    return any(snip in c["text"].lower() for c in chunks)


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else {}


def judge(question: str, context: str, expected: str, generated: str) -> dict:
    """LLM-as-judge: rate groundedness and correctness of the generated answer."""
    prompt = f"""You are grading a retrieval-augmented (RAG) system's answer.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

REFERENCE ANSWER (ground truth): {expected}

SYSTEM ANSWER: {generated}

Judge two things:
- grounded: Is EVERY factual claim in the system answer supported by the retrieved context? true/false
- correct:  Does the system answer convey the same key facts as the reference answer? true/false

Respond with ONLY this JSON, nothing else:
{{"grounded": true/false, "correct": true/false, "reason": "<one short sentence>"}}"""
    resp = client.messages.create(
        model=JUDGE_MODEL, max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        v = _parse_json(resp.content[0].text)
        return {"grounded": bool(v.get("grounded")), "correct": bool(v.get("correct")),
                "reason": v.get("reason", "")}
    except Exception as e:
        return {"grounded": False, "correct": False, "reason": f"judge parse error: {e}"}


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------
def run():
    pipe_answer = _demo_answer if DEMO else answer
    total = len(TEST_SET)
    hits = grounded = correct = 0

    print(f"\n=== AskMyDocs eval — run '{RUN_LABEL}' — {total} questions — k={TOP_K} ===\n")
    for i, case in enumerate(TEST_SET, 1):
        q = case["question"]
        gen, chunks = pipe_answer(q)
        hit = retrieval_hit(chunks, case["expected_snippet"])
        context = "\n\n".join(c["text"] for c in chunks)
        v = judge(q, context, case["expected_answer"], gen)

        hits += hit
        grounded += v["grounded"]
        correct += v["correct"]

        flag = lambda b: "PASS" if b else "FAIL"
        print(f"[{i:>2}/{total}] retrieval:{flag(hit)}  grounded:{flag(v['grounded'])}  "
              f"correct:{flag(v['correct'])}  | {q}")
        if not (hit and v["grounded"] and v["correct"]):
            print(f"        ↳ {v['reason']}")

    pct = lambda n: round(100 * n / total, 1)
    summary = {
        "run": RUN_LABEL,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n": total,
        "retrieval_hit_rate": pct(hits),
        "faithfulness": pct(grounded),
        "correctness": pct(correct),
    }
    print("\n--- SUMMARY ------------------------------------------")
    print(f"  Retrieval hit-rate (recall@{TOP_K}): {summary['retrieval_hit_rate']}%")
    print(f"  Faithfulness (grounded):           {summary['faithfulness']}%")
    print(f"  Correctness (vs reference):        {summary['correctness']}%")
    print("------------------------------------------------------")

    with open("eval_results.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print("Appended to eval_results.jsonl — change one variable and re-run to compare.\n")


if __name__ == "__main__":
    run()