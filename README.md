# Clarification Codegen

EIG-based clarification pipeline for code generation using `Qwen/Qwen2.5-Coder-1.5B-Instruct` on the MBPP dataset.

## Setup

Use Python 3.11, then install the project dependencies:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run `eig_pipeline.ipynb` to generate candidate solutions, ask clarification questions, and evaluate pass@1. Downloaded data, model artifacts, and run results are kept out of version control.
