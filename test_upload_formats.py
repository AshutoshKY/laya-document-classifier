"""End-to-end file-format tests for the classifier app.

Needs the app running first:  python doc_classifier_app.py   (port 8420)
Run with the project venv python (needs Pillow, pymupdf, openpyxl).

Generates fixtures in a temp dir — a receipt photo (PNG), an image-only
scanned boarding-pass PDF, and a synthetic .xlsx — and uploads each.
"""
import json, os, sys, tempfile, urllib.request

from PIL import Image, ImageDraw, ImageFont
import pymupdf
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
probes = json.load(open(os.path.join(HERE, "probes.json")))
SCRATCH = tempfile.mkdtemp(prefix="laya_fmt_")

def text_image(text, path):
    lines = text.split("\n")
    font = ImageFont.load_default(16)
    w = max(font.getlength(l) for l in lines) + 40
    h = len(lines) * 24 + 40
    img = Image.new("RGB", (int(w), int(h)), "white")
    d = ImageDraw.Draw(img)
    for i, l in enumerate(lines):
        d.text((20, 20 + i * 24), l, fill="black", font=font)
    img.save(path)

# 1) receipt photo (PNG)
png_path = os.path.join(SCRATCH, "receipt_photo.png")
text_image(probes["receipt"], png_path)

# 2) scanned boarding pass (image-only PDF)
pdf_path = os.path.join(SCRATCH, "boarding_pass_scan.pdf")
doc = pymupdf.open()
page = doc.new_page()
text_image(probes["boarding_pass"], os.path.join(SCRATCH, "bp_tmp.png"))
page.insert_image(pymupdf.Rect(50, 50, 500, 400), filename=os.path.join(SCRATCH, "bp_tmp.png"))
doc.save(pdf_path)
doc.close()

# 3) synthetic xlsx (tracker-style tabular data)
xlsx_path = os.path.join(SCRATCH, "habit_tracker.xlsx")
wb = openpyxl.Workbook()
ws = wb.active
ws.append(["Date", "Habit", "Completed", "Duration_min", "Notes"])
for i, (habit, mins) in enumerate([("Reading", 30), ("Workout", 45), ("Meditation", 15),
                                   ("Reading", 25), ("Workout", 50)], start=1):
    ws.append([f"2026-09-{i:02d}", habit, True, mins, ""])
wb.save(xlsx_path)

BASE = os.environ.get("LAYA_CLASSIFIER_URL", "http://127.0.0.1:8420")
def post_file(path, pack="general"):
    boundary = "----pybnd"
    fn = os.path.basename(path)
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="pack"\r\n\r\n{pack}\r\n'.encode(),
             f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{fn}"\r\n\r\n'.encode()
             + open(path, "rb").read() + b"\r\n",
             f"--{boundary}--\r\n".encode()]
    req = urllib.request.Request(BASE + "/classify", data=b"".join(parts),
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())

tests = [(png_path, "general", "receipt"), (pdf_path, "general", "travel_ticket"),
         (pdf_path, "everyday life", "boarding_pass"), (xlsx_path, "general", "spreadsheet")]
ok = 0
for path, pack, expect in tests:
    j = post_file(path, pack)
    good = j.get("type") == expect
    ok += good
    print(f"{'OK ' if good else 'MISS'} {os.path.basename(path):24s} [{pack}] -> {j.get('type')} {j.get('confidence', 0):.2f} (want {expect})")
print(f"{ok}/{len(tests)}")
sys.exit(0 if ok == len(tests) else 1)
