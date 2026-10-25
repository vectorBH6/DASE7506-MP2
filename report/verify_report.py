"""Check report evidence, code snippets, page structure and immutable model sources."""
from pathlib import Path
import json,re,hashlib,subprocess,sys
from datetime import datetime,timezone
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/"report";CODE=ROOT/"code"
d=json.loads((HERE/"results.json").read_text())
pdf=HERE/"MP2_Final_Report.pdf"
reader=PdfReader(str(pdf))
assert len(reader.pages)==8
texts=[page.extract_text() for page in reader.pages]
for i,text in enumerate(texts,1):
    assert f"{i} / 8" in text
    assert "3036705979" in text
    assert "li shanghang" in text
    assert "\u25a0" not in text and "\ufffd" not in text
    assert len(text)>1500
alltext="\n".join(texts)
for needle in ("393/467","84.1541755888651","2,112","29,766","1.166","L2","75.00%"):
    assert needle in alltext,needle
for needle in ("Model SHA256: "+d["checkpoint_hashes"]["model.npz"],
               "Config SHA256: "+d["checkpoint_hashes"]["config.json"],
               "Code freeze: "+d["freeze_id"]):
    assert needle in alltext,needle
assert d["models"]["final"]["val_correct"]==751
assert d["models"]["final"]["conflict_correct"]==308
assert d["models"]["final"]["test"]["correct"]==393
assert sum(r["final_correct"] for r in d["strata"])==751
assert sum(r["n"] for r in d["calibration"]["bins"])==355
for name,r in d["models"].items():
    assert abs(r["test"]["accuracy"]-100*r["test"]["correct"]/r["test"]["n"])<1e-10
    assert r["test"]["n"]==467 and r["val_n"]==798
paired=d["paired_validation"]["residual"]
assert (paired["corrected"],paired["regressed"],paired["net"])==(21,4,17)
md=(HERE/"REPORT.md").read_text()
for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)",md):
    assert (HERE/target).is_file(),target
commands=re.findall(r"~~~bash\n(.*?)~~~",md,re.S)
for command in commands:
    run=subprocess.run(["bash","-n"],input=command,text=True,capture_output=True)
    assert run.returncode==0,run.stderr
for target in re.findall(r"\]\(([^)]+)\)",(ROOT/"README.md").read_text()):
    assert (ROOT/target).exists(),target
frozen=json.loads((CODE/"solution/runs/handcrafted_extended_v2/freeze.json").read_text())
assert all(hashlib.sha256((CODE/name).read_bytes()).hexdigest()==value for name,value in frozen["sha256"].items())
assert hashlib.sha256((CODE/"solution/runs/handcrafted_extended_v2/source.tar").read_bytes()).hexdigest()==frozen["source_archive_sha256"]
official=CODE/"solution/runs/handcrafted_extended_v2/test/score.json"
assert json.loads(official.read_text())==d["models"]["final"]["test"]
reviewed_sha="01051155fa20a6e14119674c28f4fb8b024747d2f31347f0fc630354d0dbd849"
visually_reviewed=hashlib.sha256(pdf.read_bytes()).hexdigest()==reviewed_sha
proof={"created_utc":datetime.now(timezone.utc).isoformat(),"pages":8,
  "all_pages_rendered_and_visually_reviewed":visually_reviewed,
  "visual_review":"All eight 100-dpi PNG renders inspected; no clipping, overlap, broken headers or table cells." if visually_reviewed else "PDF changed since manual visual review; render and inspect again.",
  "page_numbers_and_identity_checked":True,"numeric_record_checks_passed":True,
  "bash_blocks_syntax_checked":len(commands),"source_and_image_links_checked":True,
  "frozen_code_and_weights_files_verified":len(frozen["sha256"]),
  "original_frozen_source_archive_verified":True,"official_score_unchanged":True,
  "test_references_used_for_report_analysis":False,
  "report_sha256":hashlib.sha256(pdf.read_bytes()).hexdigest(),
  "scope":"Reporting only; no retraining, hyperparameter search or new test selection."}
(HERE/"verification.json").write_text(json.dumps(proof,indent=2)+"\n")
print(json.dumps(proof,indent=2))
