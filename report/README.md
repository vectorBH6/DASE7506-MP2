# Final report

**Author:** li shanghang  
**Student ID:** 3036705979

The submission report is **MP2_Final_Report.pdf** (8 pages, below the 10-page limit).

- **REPORT.md**: editable English manuscript export. PDF text blocks live in generate_report.py; rebuilding regenerates this Markdown file.
- **figures/**: algorithm diagram and experiment/analysis plots in PNG/SVG.
- **results.json**: archived results, descriptive validation diagnostics and checkpoint hashes.
- **analyze_results.py**: rebuilds the data summary from existing artifacts and train/validation references.
- **generate_report.py**: renders figures and produces the PDF and manuscript.
- **verify_report.py**: checks numerical evidence, command syntax and immutable model hashes. Its visual-review flag applies to the PDF hash actually inspected.
- **verification.json**: numerical, source-hash, page-count and rendering checks.
- **manifest.json**: hashes of the delivered report files.

The report covers the algorithm, representation and objective, evaluation protocol, baselines, feature ablations, unsuccessful experiments, paired validation analysis, confidence, failure cases, resources, reproduction, limitations and AI/reuse disclosure. Report analysis does not train or select a new model and does not read test references.

## Rebuild

From the repository root, with Python 3.12 and the original experiment artifacts present:

~~~bash
python3 -m venv report/.venv
report/.venv/bin/python -m pip install -r report/requirements.txt
report/.venv/bin/python report/analyze_results.py
report/.venv/bin/python report/generate_report.py
~~~

The PDF renderer uses the DejaVu fonts at /usr/share/fonts/truetype/dejavu; adjust FONTDIR for another platform. Poppler (pdftoppm, pdfinfo) is useful for visual checking but is not required to open the final PDF. Minor image/layout differences across platforms are possible.

The final method remains code/solution/residual_ranker/extended_submission.py with its matching lexical checkpoint. The report and its figures are documentation, not inference assets. The original frozen source and weights remain unchanged. Public code/checkpoint release and the course submission are separate steps.

Data attribution and retained license are copied to DATA_NOTICE.txt and DATA_LICENSE.txt for report-source redistribution.
