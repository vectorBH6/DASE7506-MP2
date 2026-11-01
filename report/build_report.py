"""Build English figures and a PDF from saved experiments; execute no models.

From report/: env -u PYTHONPATH ../code/solution/.venv/bin/python build_report.py
The manuscript is report.md; evidence.json records source paths and SHA-256 hashes.
"""
import os
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = tempfile.TemporaryDirectory(prefix=".build-", dir=HERE)
os.environ["MPLCONFIGDIR"] = BUILD.name
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"

import hashlib
from html import escape
from importlib.metadata import version
import json
import re
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle,
)

BLUE, ORANGE, GRAY = "#245c89", "#c47b24", "#6e7883"
SOURCES = {}


def read(relative):
    path = ROOT / "code/solution/runs" / relative
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def collect():
    """读取原始证据；原融合为最终方案，注意力只保留探索性比较。"""
    p3 = read("p3/validation/summary.json")
    latest = read("tree_v2/validation/summary.json")
    calibration = read("calibrated_ensemble_v1/validation/summary.json")
    definitions = [
        ("baseline", "Original voting", None), ("embedding", "Embedding only", None),
        ("b1", "B1 tie centrality", None), ("b2", "B2 group centrality", None),
        ("linear", "L0 basic linear", "linear_v1"), ("mlp", "M0 basic MLP", "mlp_v1"),
        ("m1", "M1 text MLP", "mlp_text_v1"), ("m2", "M2 arithmetic MLP", "mlp_arithmetic_v1"),
        ("linear_text", "L1 text linear", "linear_text_v1"),
        ("ensemble", "Final L0 + M1", "ensemble_v1"),
        ("calibrated", "Calibrated L0 + M1", "calibrated_ensemble_v1"),
        ("tree_300", "Tree 300", "tree_v1"), ("tree", "Tree 1000", "tree_v2"),
    ]
    methods = {}
    for key, label, directory in definitions:
        record = (read("tree_v1/validation/summary.json")["methods"]["tree"] if key == "tree_300"
                  else (p3 if key in p3["methods"] else latest)["methods"][key])
        item = {"label": label, "validation": record["official"], "conflict": record["conflict"],
                "oof": record.get("train_oof", {}).get("overall")}
        if key == "baseline":
            item["test"] = read("baselines/final-test-score.json")
        if directory:
            item["test"] = read(directory + "/test/score.json")
        methods[key] = item
    ablations = {name: {"official": p3["methods"]["m1_" + name]["official"],
                       "comparison": p3["methods"]["m1_" + name]["vs_m1"]}
                 for name in ("no_votes", "no_embedding", "no_context", "no_style", "pairwise")}
    attention = read("attention_v1/summary.json")
    attention["validation"] = read("attention_v1/validation_summary.json")
    attention["plan"] = read("attention_v1/plan.json")
    attention["checks"] = read("attention_v1/checks.json")
    attention["post_score_checks"] = read("attention_v1/post_score_checks.json")
    attention["training"] = read("attention_v1/training_summary.json")
    attention["fits"] = {name: read(f"attention_v1/{name}/training.json") for name in ("gated", "attention")}
    for name in attention["methods"]:
        if read(f"attention_v1/{name}/test/score.json") != attention["methods"][name]["official"]:
            raise ValueError("注意力汇总与官方分数不一致")
        if name.endswith("ensemble"):
            attention["methods"][name]["preparation"] = read(f"attention_v1/{name}/preparation.json")

    final_test = read("ensemble_v1/test/summary.json")
    final_validation = read("ensemble_v1/validation/summary.json")
    freeze = read("ensemble_v1/test/freeze.json")
    archive_path = ROOT / "code/solution/runs/ensemble_v1/test/source.tar"
    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if archive_hash != freeze["source_archive"]["sha256"]:
        raise ValueError("原融合源码快照哈希不符")
    SOURCES[str(archive_path.relative_to(ROOT))] = archive_hash
    # 后续重构不改写旧冻结身份：源码检查快照，其余检查原文件。
    with tarfile.open(archive_path) as archive:
        archived = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                    for m in archive.getmembers() if m.isfile()}
    invalid = [name for name, digest in freeze["sha256"].items()
               if (archived[name] if name in archived else hashlib.sha256((ROOT / name).read_bytes()).hexdigest()) != digest]
    if invalid:
        raise ValueError(f"原冻结快照或资产不匹配: {invalid}")
    current_differences = [name for name, digest in freeze["sha256"].items()
                          if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    checkpoint = ROOT / "code/solution/runs/ensemble_v1/checkpoint"
    assets = {str(p.relative_to(checkpoint)): p.stat().st_size
              for p in sorted(checkpoint.rglob("*")) if p.is_file()}
    if sum(assets.values()) != final_test["inference_asset_bytes"]:
        raise ValueError("最终完整资产大小与原记录不一致")
    if methods["ensemble"]["test"] != final_test["official"]:
        raise ValueError("最终汇总与官方评分不一致")
    for name, digest in final_test["artifact_sha256"].items():
        if hashlib.sha256((archive_path.parent / name).read_bytes()).hexdigest() != digest:
            raise ValueError("原预测或官方评分文件已改变")

    result = {
        "report_date": "2026-10-21", "student": {"name": "Li Shanghang", "id": "3036705979"},
        "final_designation": {
            "method": "L0 + M1", "run": "ensemble_v1",
            "basis": "Restore the original saved validation preference; withdraw the temporary test-based attention designation.",
            "chronology": "Earlier test results were already seen during extensions; restoring the choice does not establish a new blind test.",
            "historical_validation_selection": attention["validation"]["selection"],
            "historical_records_rewritten": False,
        },
        "methods": methods, "ablations": ablations, "attention": attention,
        "final_test": final_test, "final_validation": final_validation,
        "final_preparation": read("ensemble_v1/preparation.json"),
        "final_checks": read("ensemble_v1/checks.json"),
        "final_freeze": freeze, "frozen_archive_and_assets_verified": True,
        "current_source_differences_from_original_freeze": current_differences,
        "final_asset_files": assets,
        "linear_training": read("linear_v1/training.json"),
        "m0_training": read("mlp_v1/training.json"),
        "m1_training": read("mlp_text_v1/training.json"),
        "m1_test": read("mlp_text_v1/test/summary.json"),
        "calibration": {name: calibration["methods"][name]["confidence"]
                        for name in ("m1", "ensemble", "calibrated")},
        "capacity": read("mlp_capacity_v1/summary.json"),
        "tree_budget_comparison": read("tree_v2/checks.json")["validation_vs_tree_v1"],
        "source_sha256": SOURCES,
        "render_environment": {name: version(name) for name in ("numpy", "matplotlib", "reportlab", "pypdf")},
        "scope": "Saved records only; no model fitting, prediction, scoring or model-test execution.",
    }
    for relative in ("guide/GUIDE.md", "code/solution/attention_ranker/model.py",
                     "code/solution/attention_ranker/train.py", "code/solution/attention_ranker/submission.py",
                     "code/solution/group_ranker/features.py", "code/solution/group_ranker/evidence.py",
                     "code/solution/group_ranker/train_linear.py", "code/solution/group_ranker/train_mlp.py",
                     "code/solution/group_ranker/mlp.py", "code/solution/group_ranker/submission_ensemble.py"):
        SOURCES[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    return result


def plot_figures(data):
    directory = HERE / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "savefig.dpi": 300})

    def save(fig, name):
        fig.savefig(directory / name, bbox_inches="tight", facecolor="white", pad_inches=.12)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 2.65))
    ax.set(xlim=(0, 12.2), ylim=(0, 3.3))
    ax.axis("off")
    boxes = [(0.08, 1.03, 1.55, 1.2, "Question\n4 candidates\nFixed vectors"),
             (1.95, 1.03, 1.6, 1.2, "Official parser\nGroup equal\nanswers"),
             (3.9, 1.03, 1.55, 1.2, "Group\nfeatures and\ncontext"),
             (5.9, 2.12, 2.25, .86, "L0: basic features\nLinear group scores"),
             (5.9, .28, 2.25, 1.02, "M1: text features\nReLU hidden layers\n64 -> 16 -> 1"),
             (8.65, 1.03, 1.65, 1.2, "Per-model\nsoftmax\nMean 0.5/0.5"),
             (10.66, 1.03, 1.45, 1.2, "Select group\nReturn index")]
    for i, (x, y, w, h, label) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=.07",
                                   facecolor="#edf3f8" if i != 5 else "#fff1df", edgecolor=BLUE, linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.2)
    for start, end in [((1.63, 1.63), (1.95, 1.63)), ((3.55, 1.63), (3.9, 1.63)),
                       ((5.45, 1.63), (5.9, 2.55)), ((5.45, 1.63), (5.9, .79)),
                       ((8.15, 2.55), (8.65, 1.83)), ((8.15, .79), (8.65, 1.43)),
                       ((10.3, 1.63), (10.66, 1.63))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color=GRAY))
    ax.text(.1, .02, "One valid group: return directly. Shared features; separate train-fitted member scalers.",
            fontsize=8.1, color=GRAY)
    save(fig, "01_pipeline.png")

    fig, ax = plt.subplots(figsize=(7.4, 1.95), layout="constrained")
    pairs = [("vs_linear", "Final ensemble vs L0"), ("vs_m1", "Final ensemble vs M1")]
    for row, (name, label) in enumerate(pairs):
        item = data["final_test"]["analysis"][name]
        change, (low, high) = item["difference_pp"], item["paired_bootstrap_95ci_pp"]
        ax.errorbar(change, row, xerr=[[change-low], [high-change]], fmt="o", capsize=3,
                    color=BLUE if row == 0 else ORANGE, linewidth=1.5)
    ax.axvline(0, color=GRAY, linestyle="--", linewidth=1)
    ax.set(yticks=range(2), yticklabels=[p[1] for p in pairs], xlim=(-2.3, 3.3),
           xlabel="Accuracy difference (percentage points)",
           title="Final fusion: paired test differences; intervals cross zero")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.15)
    save(fig, "02_ensemble_comparison.png")

    fig, ax = plt.subplots(figsize=(7.4, 2.65), layout="constrained")
    groups = data["m1_test"]["analysis"]["by_group_count"]
    entries = [(-.24, "L0", {str(g): groups[str(g)]["frozen_linear"] for g in (2,3,4)}, GRAY),
               (0, "M1", {str(g): groups[str(g)]["m1"] for g in (2,3,4)}, ORANGE),
               (.24, "Final L0 + M1", data["final_test"]["analysis"]["by_group_count"], BLUE)]
    for shift, label, values, color in entries:
        values = [values[str(g)] for g in (2, 3, 4)]
        bars = ax.bar(np.arange(3) + shift, [v["accuracy"] for v in values], width=.22, label=label, color=color)
        ax.bar_label(bars, labels=[f"{v['correct']}/{v['n']}" for v in values], padding=3, fontsize=8)
    ax.set(xticks=range(3), xticklabels=["2 groups (n=303)", "3 groups (n=126)", "4 groups (n=38)"],
           ylim=(0, 112), ylabel="Accuracy (%)", title="Frozen test: disagreement remains difficult")
    ax.legend(frameon=False, ncols=3, loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    save(fig, "03_group_difficulty.png")

    fig, ax = plt.subplots(figsize=(7.4, 2.85), layout="constrained")
    ax.plot([.35, 1], [.35, 1], "--", color=GRAY, linewidth=1, label="Confidence = accuracy")
    for name, label, color in (("m1", "M1 alone", ORANGE), ("ensemble", "Final L0 + M1", BLUE)):
        bins = data["calibration"][name]["conflict_confidence_bins"]
        xs, ys = [b["mean_confidence"] for b in bins], [b["accuracy"] / 100 for b in bins]
        ax.plot(xs, ys, "o-", color=color, label=label, linewidth=1.6)
        if name == "ensemble":
            for x, y, bucket in zip(xs, ys, bins):
                ax.annotate(f"n={bucket['n']}", (x, y), xytext=(0, 9), ha="center", textcoords="offset points", fontsize=8)
    ax.set(xlim=(.35, 1.03), ylim=(.35, 1.06), xlabel="Mean maximum group probability", ylabel="Observed accuracy",
           title="Validation reliability on conflicting questions (n=355)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(alpha=.15)
    save(fig, "04_confidence.png")


def render_pdf():
    fonts = Path("/usr/share/fonts/truetype/dejavu")
    for name, file in (("Latin", "DejaVuSans.ttf"), ("LatinBold", "DejaVuSans-Bold.ttf"),
                       ("Mono", "DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
    pdfmetrics.registerFontFamily("Latin", normal="Latin", bold="LatinBold", italic="Latin", boldItalic="LatinBold")
    styles = {
        "body": ParagraphStyle("body", fontName="Latin", fontSize=9.1, leading=13.2, spaceAfter=5),
        "title": ParagraphStyle("title", fontName="LatinBold", fontSize=18, leading=24, spaceAfter=10, textColor=colors.HexColor(BLUE)),
        "h2": ParagraphStyle("h2", fontName="LatinBold", fontSize=12.5, leading=18.5, spaceBefore=7, spaceAfter=6, keepWithNext=True, textColor=colors.HexColor(BLUE)),
        "h3": ParagraphStyle("h3", fontName="LatinBold", fontSize=10.1, leading=15.5, spaceBefore=6, spaceAfter=4, keepWithNext=True),
        "table": ParagraphStyle("table", fontName="Latin", fontSize=8, leading=11.4),
        "reference": ParagraphStyle("reference", fontName="Latin", fontSize=8.3, leading=11.8, spaceAfter=3),
        "caption": ParagraphStyle("caption", fontName="Latin", fontSize=8, leading=11.5, spaceAfter=7, textColor=colors.HexColor(GRAY)),
        "code": ParagraphStyle("code", fontName="Mono", fontSize=8.2, leading=12.5, spaceBefore=3, spaceAfter=7, backColor=colors.HexColor("#f2f5f8")),
    }

    def markup(text):
        def inline(value):
            value = escape(value)
            value = re.sub(r"`([^`]+)`", r'<font name="Mono" size="8">\1</font>', value)
            return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)

        parts, end = [], 0
        for match in re.finditer(r"\[([^]]+)\]\(([^)]+)\)", text):
            parts.append(inline(text[end:match.start()]))
            parts.append(f'<link href="{escape(match[2], quote=True)}" color="{BLUE}">{inline(match[1])}</link>')
            end = match.end()
        parts.append(inline(text[end:]))
        return "".join(parts)

    def paragraph(text, kind="body"):
        return Paragraph(markup(text), styles[kind])

    width = A4[0] - 84
    story, lines, index = [], (HERE / "report.md").read_text(encoding="utf-8").splitlines(), 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line:
            continue
        if line == "<!-- PAGEBREAK -->":
            story.append(PageBreak())
        elif line.startswith("!["):
            match = re.match(r"!\[([^]]*)\]\(([^)]+)\)", line)
            image = Image(str(HERE / match[2]))
            image.drawHeight *= width / image.drawWidth
            image.drawWidth = width
            story.extend((image, Spacer(1, 4)))
        elif line.startswith("```"):
            content = []
            while index < len(lines) and not lines[index].startswith("```"):
                content.append(lines[index])
                index += 1
            index += 1
            story.append(Preformatted("\n".join(content), styles["code"]))
        elif line.startswith("|"):
            rows = [line]
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(lines[index].strip())
                index += 1
            cells = [[cell.strip() for cell in row.strip("|").split("|")] for row in rows]
            cells = [row for row in cells if not all(re.fullmatch(r":?-+:?", cell) for cell in row)]
            count = len(cells[0])
            if any(len(row) != count for row in cells):
                raise ValueError("Inconsistent Markdown table columns; avoid literal pipes inside cells")
            ratios = ([.24, .76] if count == 2 else [.38, .31, .31] if count == 3
                      else [.31, .23, .23, .23] if count == 4 else [1/count]*count)
            if cells[0][0] == "Feature block":
                ratios = [.25, .12, .63]
            elif cells[0][0] == "Component / measurement":
                ratios = [.34, .66]
            elif cells[0][0] == "Question and reference calculation":
                ratios = [.68, .16, .16]
            elif count == 4 and cells[0][1] == "Recorded training / preparation":
                ratios = [.19, .43, .21, .17]
            table = Table([[paragraph(cell, "table") for cell in row] for row in cells],
                          colWidths=[width * ratio for ratio in ratios], repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf0f6")),
                ("LINEBELOW", (0, 0), (-1, 0), .8, colors.HexColor(BLUE)),
                ("LINEBELOW", (0, 1), (-1, -1), .35, colors.HexColor("#dce1e7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
            story.extend((table, Spacer(1, 8)))
        elif line.startswith("### "):
            story.append(paragraph(line[4:], "h3"))
        elif line.startswith("## "):
            story.append(paragraph(line[3:], "h2"))
        elif line.startswith("# "):
            story.append(paragraph(line[2:], "title"))
        else:
            kind = "caption" if line.startswith("Figure ") else "reference" if re.match(r"^\[(?:[0-9]+|M2)\]", line) else "body"
            story.append(paragraph(line, kind))

    def decorate(canvas, document):
        canvas.saveState()
        canvas.setFont("Latin", 8)
        canvas.setFillColor(colors.HexColor(GRAY))
        canvas.drawString(42, A4[1] - 24, "DASE7506 MP2 | Li Shanghang | 3036705979")
        canvas.drawRightString(A4[0] - 42, 22, str(document.page))
        canvas.setStrokeColor(colors.HexColor("#dce1e7"))
        canvas.line(42, A4[1] - 30, A4[0] - 42, A4[1] - 30)
        canvas.restoreState()

    output = HERE / "report.pdf"
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=42, leftMargin=42,
                            topMargin=42, bottomMargin=40, title="DASE7506 MP2 - Linear and MLP Group Ranking",
                            author="Li Shanghang (3036705979)")
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    reader = PdfReader(output)
    if len(reader.pages) > 10:
        raise ValueError(f"Report exceeds 10 pages: {len(reader.pages)}")
    return len(reader.pages)


def main():
    data = collect()
    (HERE / "evidence.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot_figures(data)
    pages = render_pdf()
    print(json.dumps({"pdf": str(HERE / "report.pdf"), "pages": pages, "figures": 4,
                      "source_records": len(SOURCES), "model_execution": False}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    finally:
        BUILD.cleanup()
