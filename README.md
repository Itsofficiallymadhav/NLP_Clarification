# Clarification Codegen

Baseline code-generation evaluation using `Qwen/Qwen2.5-Coder-1.5B-Instruct`
on the MBPP dataset.

## Setup

Use Python 3.11, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run `notebooks/baseline.ipynb` to download the dataset, generate solutions, and
evaluate the baseline. Downloaded data, model artifacts, and run results are
kept out of version control.
