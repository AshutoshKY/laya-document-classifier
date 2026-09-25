"""End-to-end verification for the classifier app.

Needs the app running first:  python doc_classifier_app.py   (port 8420)

Runs the 10-genre suite against the general pack, the focused-pack probes,
and — optionally — your own files, passed as path=expected_type pairs:

    python verify_packs.py ~/Downloads/resume.pdf=resume ~/Downloads/tracker.csv=spreadsheet
"""
import json, os, sys, time, urllib.request

BASE = os.environ.get("LAYA_CLASSIFIER_URL", "http://127.0.0.1:8420")
HERE = os.path.dirname(os.path.abspath(__file__))
probes = json.load(open(os.path.join(HERE, "probes.json")))

for _ in range(90):
    try:
        urllib.request.urlopen(BASE + "/", timeout=2)
        break
    except Exception:
        time.sleep(1)

def post(fields, file_path=None):
    boundary = "----pybnd"
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    if file_path:
        fn = os.path.basename(file_path)
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{fn}"\r\n\r\n'.encode()
                     + open(file_path, "rb").read() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(BASE + "/classify", data=b"".join(parts),
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())

PACK_OF = {"chat_transcript": "correspondence", "exam": "education & research",
           "blog_post": "media & publishing", "newsletter": "media & publishing",
           "presentation": "work & tech ops"}

g10 = ["resume", "research_paper", "news_article", "invoice", "manual", "book",
       "recipe", "magazine", "legal_contract", "business_report"]
ok = 0
for n in g10:
    j = post({"text": probes[n], "pack": "general"})
    ok += j["type"] == n
    if j["type"] != n:
        print(f"MISS(general) {n} -> {j['type']} {j['confidence']:.2f}")
print(f"general pack / 10-genre suite: {ok}/10")

ok = 0
for n, p in PACK_OF.items():
    j = post({"text": probes[n], "pack": p})
    ok += j["type"] == n
    print(f"{'OK ' if j['type']==n else 'MISS'}({p}) {n:16s} -> {j['type']:16s} {j['confidence']:.2f}")

user_files = []
for arg in sys.argv[1:]:
    if "=" not in arg:
        sys.exit(f"bad argument {arg!r}: expected path=expected_type")
    path, expect = arg.rsplit("=", 1)
    user_files.append((os.path.expanduser(path), expect))

if user_files:
    ok = 0
    for f, expect in user_files:
        j = post({"pack": "general"}, file_path=f)
        ok += j["type"] == expect
        print(f"{'OK ' if j['type']==expect else 'MISS'} {os.path.basename(f):46s} -> {j['type']:16s} {j['confidence']:.2f}")
    print(f"user files: {ok}/{len(user_files)}")
