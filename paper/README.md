# Adaptive AI Interview Research Report

The PDF is the presentation-ready 20-page report. `adaptive_ai_interview_report.tex` is the editable LaTeX source, written for the IEEE Access class included in this folder. `build_long_report.py` rebuilds the paginated PDF and LaTeX source from the same report text. The benchmark figures are copied from `reports/seeded_benchmark`; the report uses only that paired seeded benchmark and does not merge older result protocols.

From the application root, rebuild the report with:

```powershell
.\.venv\Scripts\python.exe paper\build_long_report.py
```

To compile the editable LaTeX source on Windows, install a LaTeX distribution, open PowerShell in this folder, then run:

```powershell
pdflatex adaptive_ai_interview_report.tex
pdflatex adaptive_ai_interview_report.tex
```

The report's evaluation is synthetic and exploratory. See the Methodology, Results, and Limitations sections for the training/evaluation environment mismatch and the measures that were not recorded.
