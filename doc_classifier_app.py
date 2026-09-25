"""Local document classifier UI powered by Laya.

Run:  .venv/bin/python doc_classifier_app.py
Open: http://127.0.0.1:8420

Paste text or drop any file: .txt/.md/.pdf (incl. scanned, via OCR),
.docx/.pptx/.xlsx/.csv, or an image (.png/.jpg/.webp/.tif/.bmp — OCR'd
with local tesseract). Laya answers a typed `choice` question in a single
forward pass — no text generation. Long documents are routed to
laya-multilingual with max_len=8192 (per the repo README's guidance).
"""
import io
import json
import os
from pathlib import Path

# Models live inside the project — set HF_HOME before laya/huggingface import.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / ".hf_cache"))

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from laya import Router
from laya.lang import analyse as lang_analyse

DEFAULT_TYPES = {
    # --- career & HR ---
    "resume": "a document presenting a person to employers: their name, email, phone, LinkedIn/GitHub links, job titles, employers with dates, degrees, skill lists, achievements",
    "cover_letter": "letter accompanying a job application: addressed to a hiring manager, expressing interest in a specific role",
    "job_posting": "job vacancy advertisement: role title, responsibilities, requirements, compensation, how to apply",
    "offer_letter": "employment offer letter: position, salary, start date, benefits, acceptance deadline",
    "payslip": "pay stub or salary slip: earnings, deductions, taxes withheld, net pay, pay period",
    # --- finance & commerce ---
    "invoice": "requests FUTURE payment from a customer: invoice number, 'amount due', pay-by date, terms like Net 30, line items, taxes",
    "receipt": "shows payment COMPLETED at a store or restaurant: store name header, items with prices, total paid, card type with last digits like VISA ****1234, change, 'thank you'",
    "purchase_order": "orders goods BEFORE delivery: 'P.O. #' or 'Purchase Order' header, quantities to ship, unit prices, requested delivery date",
    "bank_statement": "bank or credit card statement: account number, statement period, list of transactions, running balance",
    "tax_document": "tax return or tax form: tax authority fields, filing year, income and deduction figures",
    "insurance_document": "insurance policy or claim: policy number, coverage details, premiums, incident description",
    "cheque": "a bank check: 'Pay to the order of' payee line, amount written in words and again in digits, cheque number, MICR line with routing and account numbers, signature line",
    "business_report": "organizational performance document: executive summary, revenue and growth %, KPIs, quarter labels, margins, analysis",
    # --- legal & government ---
    "legal_contract": "legal document: agreement, contract, terms and conditions, clauses, parties",
    "court_document": "court filing or ruling: case number, plaintiff and defendant, motion, order, judge",
    "patent": "legal patent text: patent number, 'Claims' of an invention, field of invention, prior art references",
    "government_form": "official government form or application: agency name, numbered fields, declarations, official-use boxes",
    "id_document": "an identity document such as a passport, driver's license or national ID card: document number, full name, date of birth, nationality or issuing state, date of issue, expiry date",
    # --- work & operations ---
    "support_ticket": "a tracked issue: ticket ID number, status and priority fields, reporter and assignee, steps to reproduce, resolution",
    "technical_spec": "a design document for a software system: architecture, API contracts, data models, bug and feature lists, configuration, prompts for AI systems",
    "manual": "product operating instructions: numbered setup steps, how to use features, troubleshooting section",
    "meeting_notes": "meeting minutes or notes: attendees, agenda items, discussion points, action items",
    "memo": "internal memo: TO/FROM/DATE/RE header block, short official announcement",
    "shipping_document": "accompanies goods in transit: carrier tracking number like 1Z999, ship-from and ship-to addresses, packing slip, bill of lading",
    "product_catalog": "product catalog or listings: multiple products with SKUs, specifications and prices",
    # --- correspondence ---
    "email": "an email message with From:/To:/Subject: headers at the very top, a personal greeting, and a sign-off with the sender's name",
    "personal_letter": "a message addressed to a specific person with a greeting like 'Dear X' and a sign-off; correspondence between people",
    "invitation": "event invitation: occasion, host, date, venue, RSVP details",
    # --- publishing & media ---
    "news_article": "journalism: reporting of recent events, headline, byline, quotes from sources",
    "press_release": "press release: FOR IMMEDIATE RELEASE, company announcement, executive quotes, media contact",
    "newsletter": "newsletter or digest: multiple short sections or curated links, subscribe/unsubscribe boilerplate",
    "magazine": "magazine article: lifestyle, entertainment, interviews, feature stories, ads",
    "blog_post": "informal first-person web writing: personal anecdotes, casual asides, contractions, talks directly to the reader",
    "review": "review of a product, place or media: rating or score, pros and cons, verdict",
    "advertisement": "marketing flyer, brochure or promotional material: slogans, offers, call to action",
    # --- long-form & creative ---
    "book": "book or book chapter: continuous prose in full paragraphs forming a narrative or exposition, chapter headings",
    "essay": "formal prose arguing a thesis: structured evidence paragraphs, conclusion, no personal anecdotes",
    "poem": "poem: line-broken verse where each line is short and capitalized, grouped in stanzas, rhyme or meter",
    "lyrics": "song lyrics: a chorus or refrain repeated several times between verses",
    "screenplay": "screenplay or script: INT./EXT. scene headings, character names in ALL CAPS before their dialogue lines",
    "religious_text": "scripture or religious text: verses, chapters, prayers, sermons",
    # --- education & research ---
    "research_paper": "academic/scientific paper: Abstract and Introduction sections, methodology, citations like (Author, Year) or [1], References list",
    "academic_transcript": "school transcript or report card: courses, grades, GPA, credits, institution name",
    "exam": "a school test paper: a course name and MIDTERM/FINAL header, numbered questions, answer options a) b) c), points per question",
    "survey": "survey or questionnaire: questions with rating scales, checkboxes or option lists",
    "certificate": "certificate, diploma or award: recipient name, achievement, issuing authority, seal or signatures",
    # --- everyday life ---
    "recipe": "recipe or cookbook page: ingredients list and cooking steps",
    "menu": "restaurant or cafe menu: sections of dishes with descriptions and prices",
    "travel_ticket": "a travel ticket or boarding pass for a flight, train or bus: carrier name, flight or train number, booking reference or PNR code, departure and arrival points, seat or coach number",
    "boarding_pass": "an airline boarding pass or flight e-ticket: airline name, flight number like 6E-214 or UA1234, 3-letter airport codes, seat number, gate, boarding time, barcode",
    "train_ticket": "a railway ticket: train name and number, PNR or reservation code, from and to station names, coach and seat/berth like S4-32, fare class, journey date",
    "hotel_booking": "a hotel reservation confirmation: hotel name, check-in and check-out dates, room type, nightly rate, confirmation number, guest name",
    "medical_record": "patient medical document: diagnosis, prescriptions, lab results, vital signs, physician notes",
    "notes": "personal notes: fragments, bullets, reminders, unstructured jottings",
    "todo_list": "task list or checklist: action items with checkboxes or markers",
    # --- data & machines ---
    "spreadsheet": "tabular data exported from a spreadsheet app: rows of comma-separated cells with numbers, dates, TRUE/FALSE values, like a tracker or ledger",
    "source_code": "raw program code lines: import/def/class/return statements, braces, indentation, no prose sentences",
    "log_file": "application or system log where every line starts with a timestamp followed by a level like INFO/WARN/ERROR, stack traces",
    "chat_transcript": "a conversation transcript between two or more people: lines like '[09:41] Name: message', short casual exchanges",
    "presentation": "a slide deck: slide-by-slide markers like 'Slide 1:', titles with terse bullet points, speaker notes",
    "other": "anything that does not fit the above categories",
}

HEAD_CHARS = 1800  # ~450 tokens; genre signal lives in the head of a document

# Measured: this 200M zero-shot encoder separates ~15 well-chosen concrete
# categories near-perfectly (14/14), but a flat 55-way choice drops to ~68%
# and hierarchical group routing to ~32%. So the full taxonomy ships as
# focused packs — each a small concrete choice set the model handles well.
GENERAL_KEYS = [
    "resume", "research_paper", "news_article", "invoice", "manual", "book",
    "recipe", "magazine", "legal_contract", "business_report",
    "personal_letter", "spreadsheet", "technical_spec", "receipt",
    "travel_ticket", "other",
]
PACK_GROUPS = {
    "career & hiring": ["resume", "cover_letter", "job_posting", "offer_letter", "payslip"],
    "finance & commerce": ["invoice", "receipt", "purchase_order", "bank_statement",
                           "tax_document", "insurance_document", "cheque", "business_report"],
    "legal & government": ["legal_contract", "court_document", "patent", "government_form",
                           "id_document"],
    "work & tech ops": ["support_ticket", "technical_spec", "manual", "meeting_notes",
                        "memo", "shipping_document", "product_catalog", "presentation"],
    "correspondence": ["email", "personal_letter", "invitation", "chat_transcript"],
    "media & publishing": ["news_article", "press_release", "newsletter", "magazine",
                           "blog_post", "review", "advertisement"],
    "creative writing": ["book", "essay", "poem", "lyrics", "screenplay", "religious_text"],
    "education & research": ["research_paper", "academic_transcript", "exam", "survey", "certificate"],
    "everyday life": ["recipe", "menu", "boarding_pass", "train_ticket", "hotel_booking",
                      "medical_record", "notes", "todo_list"],
    "data & machine output": ["spreadsheet", "source_code", "log_file"],
}

# Measured: the invoice/receipt near-pair resolves differently depending on
# pack context. v3 wording is required in "general" (orig flips receipt ->
# invoice 0.82), but the SAME v3 wording flips receipt -> purchase_order 0.53
# in "finance & commerce", where orig wording is the only working pair.
# So wording is overridden per pack here — do not unify these.
PACK_OVERRIDES: dict = {
    "general": {
        "invoice": "requests FUTURE payment: invoice number, 'amount due', pay-by date, terms like Net 30, billed to a company",
        "receipt": "a store or restaurant bill settled at the counter: items with prices, TOTAL, payment card digits, change, cashier, 'thank you'",
    },
}

PACKS: dict = {"general": {k: DEFAULT_TYPES[k] for k in GENERAL_KEYS}}
for _name, _keys in PACK_GROUPS.items():
    # No "other" in focused packs — measured: it becomes a magnet for
    # unusual-but-valid members (chat logs, exams, blog posts all collapsed
    # into it). A pack is already domain-scoped by the user's choice.
    PACKS[_name] = {k: DEFAULT_TYPES[k] for k in _keys}
for _name, _over in PACK_OVERRIDES.items():
    PACKS[_name].update(_over)

app = FastAPI(title="Laya Document Classifier")
router = Router()


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def _ocr_pil_image(img) -> str:
    import pytesseract
    import shutil
    cmd = shutil.which("tesseract") or "/opt/homebrew/bin/tesseract"
    pytesseract.pytesseract.tesseract_cmd = cmd
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    return pytesseract.image_to_string(img)


def _ocr_image(data: bytes) -> str:
    from PIL import Image
    return _ocr_pil_image(Image.open(io.BytesIO(data)))


def _ocr_pdf(data: bytes, max_pages: int = 10) -> str:
    # Scanned PDF (no text layer): render pages and OCR them.
    import pymupdf
    from PIL import Image
    chunks = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            if len(chunks) >= max_pages:
                break
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            chunks.append(_ocr_pil_image(img))
    return "\n".join(chunks)


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if len(text.strip()) < 40:  # image-only scan: no text layer
            text = _ocr_pdf(data)
        return text
    if ext == ".docx":
        import docx
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)
    if ext == ".pptx":
        from pptx import Presentation
        prs = Presentation(io.BytesIO(data))
        slides = []
        for i, slide in enumerate(prs.slides, 1):
            lines = [sh.text_frame.text for sh in slide.shapes if sh.has_text_frame]
            slides.append(f"Slide {i}:\n" + "\n".join(lines))
        return "\n\n".join(slides)
    if ext == ".xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        rows = []
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                rows.append(", ".join("" if c is None else str(c) for c in row))
        return "\n".join(rows)
    if ext in IMG_EXTS:
        return _ocr_image(data)
    return data.decode("utf-8", errors="replace")


# Measured architecture note: a two-pass hierarchical version (abstract group ->
# subtype) scored 14/47 on probes — abstract group criteria fail on this model.
# Flat with concrete structural criteria scored 10/10 (14 cats) and 34/47 (55
# cats). Flat is the shipped design; criteria wording carries the accuracy.
def classify(text: str, criteria: dict, filename: str = "") -> dict:
    questions = {"document_type": {
        "type": "choice",
        "instructions": "What type of document is this? Judge by its overall structure and purpose.",
        "criteria": criteria,
    }}

    # Filename is strong signal for tabular data (bare CSV cells lack prose
    # signal) but measurably hurts prose documents — measured: a resume PDF
    # flips from resume 0.65 to technical_spec 0.93 when ANY filename is
    # attached. So attach it only for tabular extensions.
    use_filename = Path(filename).suffix.lower() in {".csv", ".tsv", ".xls", ".xlsx"}

    def make_state(t: str):
        return {"filename": filename, "content": t} if use_filename else t

    if len(text) > HEAD_CHARS:
        head = text[:HEAD_CHARS]
        # Multilingual only on HARD non-English evidence: non-latin script or
        # actual diacritics. Measured: a bare language guess misroutes symbol-
        # heavy Latin-script text (an env/config file "looks like pt") and the
        # multilingual checkpoint then returns a flat garbage spread (top
        # confidence 0.21). Accent-free French/Spanish prose now goes to the
        # English head instead — acceptable: it mis-grades sometimes, the
        # reverse misroute garbages every time.
        det = lang_analyse(head)
        diacritic_rate = det.get("diacritic_rate")
        use_multilingual = (not det["is_english"]) and (
            det["script"] != "latin"
            or (isinstance(diacritic_rate, (int, float)) and diacritic_rate > 0))
        if use_multilingual:
            # genuinely non-English: multilingual checkpoint reads up to 8,192 tokens
            result = router.predict(make_state(text), questions, model="multilingual", max_len=8192)
        else:
            # long English doc: the head carries the genre signal; the English
            # checkpoint beats multilingual on English prose
            result = router.predict(make_state(head), questions, model="english")
    else:
        result = router.predict(make_state(text), questions)
    ans = result["answers"]["document_type"]
    probs = ans.get("probabilities", {})
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "type": ans["choice"],
        "confidence": round(ans.get("answer_confidence", 0), 4),
        "distribution": [{"type": k, "probability": round(v, 4)} for k, v in ranked],
        "routing": {"model": result["routing"]["model"], "reason": result["routing"]["reason"]},
        "chars": len(text),
    }


@app.post("/classify")
async def classify_endpoint(
    text: str = Form(default=""),
    types: str = Form(default=""),
    pack: str = Form(default="general"),
    file: UploadFile | None = File(default=None),
):
    filename = ""
    if file is not None and file.filename:
        filename = file.filename
        data = await file.read()
        try:
            text = extract_text(filename, data)
        except Exception as e:
            return JSONResponse({"error": f"could not extract text from {filename}: {e}"}, status_code=400)
        # Measured: classifying a near-empty extraction (e.g. a photo of a
        # logo -> 35 OCR'd chars) returns confident garbage ("legal contract
        # 97%"). Refuse instead and say why.
        if len("".join(ch for ch in text if ch.isalnum())) < 40:
            return JSONResponse(
                {"error": f"could not extract enough readable text from {filename} "
                          f"(got {len(text.strip())} chars). If it's an image or scan, "
                          "use a sharper, well-lit, text-focused shot — or paste the text."},
                status_code=400)
    text = (text or "").strip()
    if not text:
        return JSONResponse({"error": "empty document"}, status_code=400)
    criteria = PACKS.get(pack, PACKS["general"])
    if types.strip():  # custom categories override the pack
        try:
            custom = json.loads(types)
            if isinstance(custom, dict) and custom:
                criteria = {str(k): str(v) for k, v in custom.items()}
        except json.JSONDecodeError:
            names = [t.strip() for t in types.split(",") if t.strip()]
            if names:
                criteria = {n: n.replace("_", " ") for n in names}
    return classify(text, criteria, filename)


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Laya — Document Classifier</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, 'SF Pro Text', system-ui, sans-serif;
         background: #0d1117; color: #e6edf3; min-height: 100vh; }
  .wrap { max-width: 860px; margin: 0 auto; padding: 40px 20px 80px; }
  h1 { font-size: 22px; font-weight: 650; margin: 0 0 4px; }
  .sub { color: #8b949e; font-size: 13px; margin-bottom: 24px; }
  .sub code { background: #161b22; padding: 1px 6px; border-radius: 5px; font-size: 12px; }
  #drop { border: 1.5px dashed #30363d; border-radius: 14px; padding: 26px;
          text-align: center; color: #8b949e; font-size: 14px; cursor: pointer;
          transition: border-color .15s, background .15s; }
  #drop.over { border-color: #58a6ff; background: #101a26; color: #c9d1d9; }
  textarea { width: 100%; min-height: 170px; margin-top: 14px; resize: vertical;
             background: #0d1117; color: #e6edf3; border: 1px solid #30363d;
             border-radius: 12px; padding: 14px; font: 13px/1.55 ui-monospace, 'SF Mono', Menlo, monospace; }
  textarea:focus { outline: none; border-color: #58a6ff; }
  .packrow { display: flex; align-items: center; gap: 10px; margin-top: 14px; font-size: 13px; color: #8b949e; }
  select { background: #0d1117; color: #e6edf3; border: 1px solid #30363d; border-radius: 9px;
           padding: 8px 12px; font-size: 13px; cursor: pointer; }
  select:focus { outline: none; border-color: #58a6ff; }
  details { margin-top: 14px; font-size: 13px; color: #8b949e; }
  details textarea { min-height: 90px; margin-top: 8px; }
  .row { display: flex; align-items: center; gap: 12px; margin-top: 16px; }
  button { background: #238636; color: #fff; border: 0; border-radius: 9px;
           padding: 10px 22px; font-size: 14px; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  #status { font-size: 13px; color: #8b949e; }
  #result { margin-top: 26px; display: none; }
  .verdict { display: flex; align-items: baseline; gap: 12px; margin-bottom: 4px; }
  .verdict .t { font-size: 26px; font-weight: 700; letter-spacing: .2px; }
  .verdict .c { font-size: 14px; color: #3fb950; font-weight: 600; }
  .meta { font-size: 12px; color: #6e7681; margin-bottom: 18px; }
  .bar-row { display: grid; grid-template-columns: 150px 1fr 58px; align-items: center;
             gap: 10px; margin: 5px 0; font-size: 13px; }
  .bar-row .name { color: #c9d1d9; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .track { background: #161b22; border-radius: 6px; height: 16px; overflow: hidden; }
  .fill { height: 100%; background: #1f6feb; border-radius: 6px; width: 0;
          transition: width .5s cubic-bezier(.2,.8,.2,1); }
  .bar-row.top .fill { background: #3fb950; }
  .bar-row .p { text-align: right; color: #8b949e; font-variant-numeric: tabular-nums; }
  .err { color: #f85149; font-size: 13px; margin-top: 14px; white-space: pre-wrap; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Laya Document Classifier</h1>
  <div class="sub">Local, single-forward-pass classification · <code>convaiinnovations/laya</code> · no data leaves this machine</div>

  <div id="drop">Drop any file here — <b>text, PDF, Word, Excel, slides, or an image</b> (photos/scans are OCR'd) — or click to browse</div>
  <input id="file" type="file" accept=".txt,.md,.markdown,.pdf,.docx,.pptx,.csv,.tsv,.xlsx,.log,.json,.html,.png,.jpg,.jpeg,.webp,.tif,.tiff,.bmp" hidden>
  <textarea id="text" placeholder="…or paste document text here"></textarea>

  <div class="packrow">
    <label for="pack">Category pack</label>
    <select id="pack">
      <option value="general" selected>General — 16 most common types</option>
      <option value="career &amp; hiring">Career &amp; hiring — resumes, job ads, offer letters, pay stubs</option>
      <option value="finance &amp; commerce">Finance &amp; commerce — invoices, receipts, POs, statements, tax, insurance</option>
      <option value="legal &amp; government">Legal &amp; government — contracts, court docs, patents, official forms</option>
      <option value="work &amp; tech ops">Work &amp; tech ops — tickets, specs, manuals, memos, slides, catalogs</option>
      <option value="correspondence">Correspondence — emails, letters, invitations, chat transcripts</option>
      <option value="media &amp; publishing">Media &amp; publishing — news, press releases, newsletters, magazines, blogs</option>
      <option value="creative writing">Creative writing — books, essays, poems, lyrics, screenplays, scripture</option>
      <option value="education &amp; research">Education &amp; research — papers, transcripts, exams, surveys, certificates</option>
      <option value="everyday life">Everyday life — recipes, menus, flight/train tickets, hotel bookings, medical records</option>
      <option value="data &amp; machine output">Data &amp; machine output — spreadsheets, source code, log files</option>
    </select>
    <span>pick the pack closest to your document for best accuracy</span>
  </div>

  <details>
    <summary>Custom categories (optional — overrides the pack)</summary>
    <textarea id="types" placeholder='Comma-separated: resume, invoice, poem, comic, thesis&#10;— or JSON: {"resume": "CV with work history", "poem": "verse, rhyme"}'></textarea>
  </details>

  <div class="row">
    <button id="go">Classify</button>
    <span id="status"></span>
  </div>

  <div id="result">
    <div class="verdict"><span class="t" id="rtype"></span><span class="c" id="rconf"></span></div>
    <div class="meta" id="rmeta"></div>
    <div id="bars"></div>
  </div>
  <div class="err" id="err"></div>
</div>

<script>
const drop = document.getElementById('drop'), fileIn = document.getElementById('file'),
      ta = document.getElementById('text'), go = document.getElementById('go'),
      status = document.getElementById('status'), err = document.getElementById('err');
let picked = null;

drop.onclick = () => fileIn.click();
fileIn.onchange = () => { if (fileIn.files[0]) setFile(fileIn.files[0]); };
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over');
                     if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); };
function setFile(f) { picked = f; drop.innerHTML = '📄 <b>' + f.name + '</b> — ' +
                      (f.size/1024).toFixed(1) + ' KB (click to change)'; }
const DROP_HTML = "Drop any file here — <b>text, PDF, Word, Excel, slides, or an image</b> (photos/scans are OCR'd) — or click to browse";
ta.oninput = () => { if (ta.value.trim()) { picked = null; fileIn.value=''; drop.innerHTML = DROP_HTML; } };

go.onclick = async () => {
  err.textContent = ''; document.getElementById('result').style.display = 'none';
  const fd = new FormData();
  if (picked) fd.append('file', picked); else fd.append('text', ta.value);
  fd.append('types', document.getElementById('types').value);
  fd.append('pack', document.getElementById('pack').value);
  go.disabled = true; status.textContent = 'classifying…';
  try {
    const res = await fetch('/classify', { method: 'POST', body: fd });
    const j = await res.json();
    if (!res.ok) { err.textContent = j.error || ('HTTP ' + res.status); return; }
    render(j);
  } catch (e) { err.textContent = String(e); }
  finally { go.disabled = false; status.textContent = ''; }
};

function render(j) {
  document.getElementById('rtype').textContent = j.type.replace(/_/g, ' ');
  document.getElementById('rconf').textContent = (j.confidence*100).toFixed(1) + '% confident';
  document.getElementById('rmeta').textContent =
    j.chars.toLocaleString() + ' chars · model: ' + j.routing.model + ' · ' + j.routing.reason;
  const bars = document.getElementById('bars'); bars.innerHTML = '';
  j.distribution.forEach((d, i) => {
    const row = document.createElement('div');
    row.className = 'bar-row' + (i === 0 ? ' top' : '');
    row.innerHTML = '<span class="name">' + d.type.replace(/_/g,' ') + '</span>' +
      '<div class="track"><div class="fill"></div></div>' +
      '<span class="p">' + (d.probability*100).toFixed(1) + '%</span>';
    bars.appendChild(row);
    requestAnimationFrame(() => requestAnimationFrame(() =>
      row.querySelector('.fill').style.width = (d.probability*100) + '%'));
  });
  document.getElementById('result').style.display = 'block';
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("LAYA_HOST", "127.0.0.1"),
                port=int(os.environ.get("LAYA_PORT", "8420")))
