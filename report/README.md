# Project report

The report source is in `main.tex`. Section content lives under `sections/`, and the bibliography is in `references.bib`.

## Build on Windows

Install either MiKTeX or TeX Live, then open a new PowerShell terminal and run:

```powershell
cd C:\Clarification-Codegen\report
.\build.ps1
```

For a clean rebuild:

```powershell
.\build.ps1 -Clean
```

The generated PDF is `report/main.pdf`. LaTeX auxiliary files are ignored by Git.

## Report data

The results table is based on the JSON artifacts in `results/`. The phase-four notebook provides the human-guided oracle evaluation mode: it asks targeted binary questions, updates candidate weights, and evaluates the selected program against the task tests.
