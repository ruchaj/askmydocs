"""
make_test_pdfs.py — generate a variety of text-based PDFs for the eval harness.

Creates test_pdfs/ with several documents spanning different domains (retail
policy, a multi-page financial report, an HR handbook, a hardware spec sheet,
and a travel FAQ) so testset.py can measure retrieval + answer quality against
real PDFs instead of a hardcoded in-memory doc.

Uses PyMuPDF (fitz), which is already a project dependency — no extra install.

Run directly:  python make_test_pdfs.py
Or import:     import make_test_pdfs; make_test_pdfs.generate()
"""

import os
from pathlib import Path

# PyMuPDF exposes itself as both `pymupdf` (canonical) and `fitz` (legacy).
# Prefer `pymupdf` — the `fitz` name can be shadowed by an unrelated package.
try:
    import pymupdf as fitz
except ImportError:
    import fitz

OUT_DIR = Path(__file__).with_name("test_pdfs")

# Each doc is a list of pages; each page is (title, [paragraphs]).
# Phrases used as expected_snippet in testset.py appear here verbatim.
DOCS = {
    "acme_returns_policy.pdf": [
        ("ACME Retail — Customer Policies", [
            "Returns and Refunds. Customers can return items within 30 days of "
            "purchase for a full refund. Items must be unused and in their "
            "original packaging.",
            "Warranty. The limited warranty covers manufacturing defects for one "
            "year. Water damage is not covered under any circumstances.",
            "Shipping. Orders over $50 qualify for free standard shipping within "
            "the United States. Expedited shipping costs $12.99 per order.",
        ]),
    ],
    "helios_financials.pdf": [
        ("Helios Corp — Annual Report 2024", [
            "Total revenue for fiscal year 2024 was $214.6 million, up 12 percent "
            "from the prior year.",
            "Operating expenses totaled $150.2 million. Net income for the year "
            "was $48.3 million.",
        ]),
        ("Helios Corp — Notes to the Accounts", [
            "Research and development spending rose to $32.4 million in 2024, "
            "reflecting continued investment in new product lines.",
            "The company took on no new debt during the year and repaid $10 "
            "million of existing loans.",
            "Helios employed 1,240 people at year end across four regional offices.",
        ]),
    ],
    "northwind_handbook.pdf": [
        ("Northwind — Employee Handbook", [
            "Paid Time Off. Full-time employees accrue 15 days of paid time off "
            "per year, in addition to 10 public holidays.",
            "Remote Work. Employees may work remotely up to three days per week "
            "with prior manager approval.",
            "Resignation. The standard notice period for resignation is two weeks "
            "for all non-management staff.",
        ]),
    ],
    "quantum_router_specs.pdf": [
        ("QuantumNet QR-500 — Technical Specifications", [
            "Wireless. The QR-500 supports Wi-Fi 6 with a maximum throughput of "
            "4.8 Gbps across both bands.",
            "Connectivity. The unit includes four gigabit Ethernet ports and one "
            "USB 3.0 port for network storage.",
            "Environment. The operating temperature range is 0 to 40 degrees "
            "Celsius at up to 90 percent humidity.",
        ]),
    ],
    "glacier_travel_faq.pdf": [
        ("Glacier Expeditions — Frequently Asked Questions", [
            "Booking. Tours can be booked up to six months in advance. A 20 "
            "percent deposit is required to confirm a reservation.",
            "Cancellations. Cancellations made more than 14 days before departure "
            "receive a full refund minus the deposit.",
            "What to Bring. Waterproof boots and layered clothing are strongly "
            "recommended for all glacier walks.",
        ]),
    ],
}


def _write_pdf(path: Path, pages: list) -> None:
    doc = fitz.open()
    for title, paragraphs in pages:
        page = doc.new_page()  # default A4
        w, h = page.rect.width, page.rect.height
        page.insert_textbox(
            fitz.Rect(60, 55, w - 60, 100),
            title, fontsize=16, fontname="hebo",
        )
        body = "\n\n".join(paragraphs)
        page.insert_textbox(
            fitz.Rect(60, 110, w - 60, h - 60),
            body, fontsize=11, fontname="helv",
        )
    doc.save(str(path))


def generate(out_dir: Path = OUT_DIR) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    for name, pages in DOCS.items():
        _write_pdf(out_dir / name, pages)
    return out_dir


if __name__ == "__main__":
    d = generate()
    n = len(list(d.glob("*.pdf")))
    print(f"Wrote {n} PDFs to {d}")
