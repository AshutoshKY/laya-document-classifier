# Laya Document Classifier

Local, zero-shot document classification built on [Laya](https://github.com/NandhaKishorM/laya)
(`convaiinnovations/laya`). Paste text or drop **any file** — text, PDF (including
scanned PDFs, OCR'd), Word, PowerPoint, Excel, CSV, or a **photo/scan image**
(png/jpg/webp/tif/bmp, OCR'd with local tesseract) — get a document type + confidence.
Everything runs on your machine; no data leaves it.

The classifier answers a typed `choice` question in a **single forward pass** through Laya's
200M encoder — no text generation, no API keys, no hosted backend.

## Quickstart

```bash
git clone https://github.com/AshutoshKY/laya-document-classifier.git
cd laya-document-classifier

python -m venv .venv && source .venv/bin/activate
pip install -e ".[serve]" pypdf python-docx python-pptx openpyxl pymupdf pillow pytesseract
brew install tesseract   # OCR engine for photos/scans (or your platform's package manager)

python doc_classifier_app.py
# open http://127.0.0.1:8420
```

First run downloads the Laya checkpoints (~2.2 GB) from Hugging Face into
`.hf_cache/` (the app pins `HF_HOME` there; it's gitignored).

## What you get

- **Web UI** (FastAPI + inline page, port 8420): paste text or drop a file, get the
  top category plus the full probability distribution as bars.
- **Any format**: `.txt/.md/.log/.json/.html/.csv/.tsv` read directly; `.pdf` via
  text-layer extraction with automatic OCR fallback for image-only scans;
  `.docx/.pptx/.xlsx` extracted natively; photos/scans (`.png/.jpg/.webp/.tif/.bmp`)
  OCR'd with local tesseract.
- **55+ document types**, shipped as focused **packs** (career, finance, legal,
  media, creative, education, everyday life, data/machine output…). Pick the pack
  closest to your document; the default General pack covers the 16 most common
  types, including receipts and flight/train tickets. Focused packs add cheques,
  boarding passes, train tickets, hotel bookings, ID documents, and more.
- **Custom categories**: comma-separated names, or `{"name": "description"}` JSON.
  Descriptions that name *structural artifacts* ("invoice: invoice number, amount
  due, Net-30 terms") beat bare genre labels. Negations ("not X") do nothing —
  it's a single forward pass, no reasoning.
- **HTTP API**: `POST /classify` with multipart form (`text` or `file`,
  optional `pack`, optional `types`) → JSON with `type`, `confidence`,
  full `distribution`, and routing info.

## Measured accuracy (this 200M zero-shot encoder)

| suite | score |
|---|---|
| General pack suite (16 types incl. receipts, travel tickets) | 13/13 |
| Real files (resume PDF, habit-tracker CSV, technical docs) | 4/4 |
| New types | boarding_pass 1.00 · train_ticket 1.00 · hotel_booking 1.00 · cheque 0.95 · id_document 0.83 |
| File-format path | receipt **photo** → receipt ✓ · scanned boarding-pass PDF → boarding_pass ✓ · .xlsx → spreadsheet ✓ |
| Focused packs | two known borderline pairs remain (newsletter↔magazine, presentation↔technical_spec) |

Why packs instead of one giant list — measured:

| design | accuracy |
|---|---|
| 16 well-separated concrete categories | 13/13 on probes |
| 55 categories in one flat choice | ~68% (near-pairs collapse) |
| hierarchical (abstract group → subtype) | ~32% (abstract group criteria fail) |

## Gotchas baked into the app

- The filename is passed to the model only for `.csv/.tsv/.xls/.xlsx` — decisive
  for tabular data, measurably hurts prose.
- Long docs classify on the first ~1,800 chars (the head carries the genre); the
  8K multilingual checkpoint is used only for genuinely non-English text.
- Confidence is a ranking signal, not a calibrated probability (the upstream
  checkpoint ships broken calibration temps).
- Criteria wording is pack-context dependent: the invoice/receipt near-pair uses
  different wording in "general" vs "finance & commerce" (measured — see
  `PACK_OVERRIDES` in the app). Don't unify them.

## Repo layout

| path | what |
|---|---|
| `doc_classifier_app.py` | the app: FastAPI server + inline UI, port 8420 |
| `laya/` | the Laya engine source (installed editable) — engine docs in `LAYA.md` |
| `probes.json` | 52 synthetic probe docs (one per category) |
| `test_quickstart.py` | reproduces the upstream quickstart end-to-end |
| `test_doc_classification.py` | 10-genre regression suite |
| `verify_packs.py` | full verification: general suite + pack probes + optional user files (needs the server on 8420) |
| `test_upload_formats.py` | file-format tests: receipt photo PNG, scanned boarding-pass PDF, .xlsx (needs the server) |

## Verification

```bash
python test_quickstart.py            # engine works end-to-end
python test_doc_classification.py    # 10-genre suite, expect 10/10
python doc_classifier_app.py &       # then, in another shell:
python verify_packs.py [path/to/file=expected_type ...]
python test_upload_formats.py        # OCR + office-format uploads, expect 4/4
```

## License & credits

Apache-2.0 (see `LICENSE`). Built on [Laya](https://github.com/NandhaKishorM/laya)
by Convai Innovations — the full engine README is preserved in `LAYA.md`.
