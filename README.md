# Laya Document Classifier

Local, zero-shot document classification built on [Laya](https://github.com/NandhaKishorM/laya)
(`convaiinnovations/laya`). Paste text or drop a file, get a document type + confidence.
Everything runs on your machine; no data leaves it.

The classifier answers a typed `choice` question in a **single forward pass** through Laya's
200M encoder — no text generation, no API keys, no hosted backend.

## Quickstart

```bash
git clone https://github.com/AshutoshKY/laya-document-classifier.git
cd laya-document-classifier

python -m venv .venv && source .venv/bin/activate
pip install -e ".[serve]" pypdf python-docx

python doc_classifier_app.py
# open http://127.0.0.1:8420
```

First run downloads the Laya checkpoints (~2.2 GB) from Hugging Face into
`.hf_cache/` (the app pins `HF_HOME` there; it's gitignored).

## What you get

- **Web UI** (FastAPI + inline page, port 8420): paste text or drop
  `.txt / .md / .pdf / .docx / .csv / .log / .json / .html`, get the top category
  plus the full probability distribution as bars.
- **50+ document types**, shipped as focused **packs** (career, finance, legal,
  media, creative, education, everyday life, data/machine output…). Pick the pack
  closest to your document; the default General pack covers the 14 most common types.
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
| General pack, 10-genre regression suite | 10/10 |
| Real files (resume PDF, habit-tracker CSV, technical docs) | 4/4 |
| Focused packs, 47 synthetic probes | 45/47 (newsletter↔magazine and presentation↔technical_spec stay borderline) |

Why packs instead of one giant list — measured:

| design | accuracy |
|---|---|
| 14 well-separated concrete categories | 14/14 |
| 55 categories in one flat choice | ~68% (near-pairs collapse) |
| hierarchical (abstract group → subtype) | ~32% (abstract group criteria fail) |

## Gotchas baked into the app

- The filename is passed to the model only for `.csv/.tsv/.xls/.xlsx` — decisive
  for tabular data, measurably hurts prose.
- Long docs classify on the first ~1,800 chars (the head carries the genre); the
  8K multilingual checkpoint is used only for genuinely non-English text.
- Confidence is a ranking signal, not a calibrated probability (the upstream
  checkpoint ships broken calibration temps).

## Repo layout

| path | what |
|---|---|
| `doc_classifier_app.py` | the app: FastAPI server + inline UI, port 8420 |
| `laya/` | the Laya engine source (installed editable) — engine docs in `LAYA.md` |
| `probes.json` | 47 synthetic probe docs (one per category) |
| `test_quickstart.py` | reproduces the upstream quickstart end-to-end |
| `test_doc_classification.py` | 10-genre regression suite |
| `verify_packs.py` | full verification: 10-genre suite + pack probes + optional user files (needs the server running on 8420) |

## Verification

```bash
python test_quickstart.py          # engine works end-to-end
python test_doc_classification.py  # 10-genre suite, expect 10/10
python doc_classifier_app.py &     # then, in another shell:
python verify_packs.py [path/to/file=expected_type ...]
```

## License & credits

Apache-2.0 (see `LICENSE`). Built on [Laya](https://github.com/NandhaKishorM/laya)
by Convai Innovations — the full engine README is preserved in `LAYA.md`.
