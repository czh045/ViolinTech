param(
    [string]$Python = "D:\Anaconda\python.exe",
    [switch]$BuildPdf
)

& $Python paper_experiments.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python plot_paper_figures.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($BuildPdf) {
    pdflatex -interaction=nonstopmode paper_draft.tex
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    pdflatex -interaction=nonstopmode paper_draft.tex
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $Python -m pytest tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
