# Clarification Codegen

This project helps AI write better code when the instructions are vague. Instead of just guessing, the system asks questions first to figure out what you actually mean. 

It's based on a paper called "20 Questions for Code". I wanted to see what happens when the AI just guesses the answer in one go versus when it asks smart questions to gather more info, using a standard set of Python problems (the MBPP dataset).

## Project status

All four parts of the project are up and running:

- **Part 1: Baseline:** Asking the AI to write the code in one try, plus a custom way to test if it works.
- **Part 2: Finding Ambiguity:** Making the AI write a few different answers for the same problem. If those answers act differently on our tests, we know the original prompt was confusing.
- **Part 3: Clarification:** A smart loop where the AI figures out the best questions to ask, updates what it knows based on your answers, and decides when it has enough info to just write the code.
- **Part 4: Human-guided Oracle:** A fully interactive version. A real person answers the AI's yes/no questions, the AI updates its choices based on those answers, and then we test the final program.

**The main result:** On 61 problems that were flagged as confusing, asking questions boosted the AI's success rate from 50.8% (guessing) to 78.7%! It only needed to ask about 1.8 questions per problem on average. You can see all the details in `notebooks/phase3.ipynb`.

## Setup

Use Python 3.11, and set up your virtual environment like this:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**Watch out for the GPU setup:** Sometimes installing PyTorch gives you the CPU-only version without warning, which makes everything super slow. Check it like this:
```python
import torch
print(torch.cuda.is_available())  # This should be True
```
If it says `False`, you need to reinstall it for your GPU (like CUDA 12.1):
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Downloading the AI:** The larger model is a 15GB download. Hugging Face will slow you way down if you aren't logged in, so just log in once before running anything:
```powershell
huggingface-cli login
```
*(Note: Please don't hardcode your login tokens in the notebooks, or they'll end up in the git history!)*

## Models

- `Qwen/Qwen2.5-Coder-1.5B-Instruct` -> I used this smaller one to test things out early on.
- `Qwen/Qwen2.5-Coder-7B-Instruct` -> This is the main one I used for the final results (I shrank it a bit so it uses less memory). I switched to this bigger model because the smaller one kept making basic coding mistakes and crashing. The bigger one writes cleaner code, making it easier to tell when it's *actually* confused by the prompt rather than just making a typo. 

## How to run

- `notebooks/baseline.ipynb` - **Part 1:** Downloads the problems, asks the AI to solve them in one try, and tests the answers.
- `notebooks/phase3.ipynb` - **Parts 2 and 3:** Generates different answers, spots the confusing parts, runs the question-asking loop, and shows the final results. Run this notebook to see the main findings.
- `notebooks/phase4_human_oracle.ipynb` - **Part 4:** Lets a person answer the clarification questions and tests the selected answer. It also includes a custom Codeforces-style problem.

The testing script is in `evaluation/evaluator.py`. It pulls the Python code out of the AI's response, runs any setup stuff, and tests the code. It also gives helpful error messages instead of just failing silently.

## The human-guided oracle example

The Part 4 notebook includes a classic competitive programming problem (`cf_1097b`). The goal is to see if you can spin a dial clockwise or counterclockwise by certain angles so that it ends up exactly where it started (a multiple of 360 degrees).

The notebook gives the AI a few basic examples like:

- `[10, 20, 30]` -> `True`
- `[10, 10, 10]` -> `False`
- `[120, 120, 120]` -> `True`

After the AI asks its questions, we test it on trickier cases it hasn't seen yet:

- `[180]` -> `False`
- `[90, 90, 90, 90]` -> `True`

This proves the question-asking loop doesn't just work on the standard dataset—you can use it for your own custom algorithmic problems and tests too!

## Project report

The formal write-up (a LaTeX report) is in the `report/` folder. It explains the main ideas from the original paper, how my version is different, the results, and the interactive Codeforces example.

The finished PDF is `report/main.pdf`. If you want to rebuild the PDF on Windows, just install MiKTeX or TeX Live and run:

```powershell
cd C:\Clarification-Codegen\report
.\build.ps1
```

For a clean rebuild:

```powershell
.\build.ps1 -Clean
```

I didn't upload the big datasets, AI models, or the final result files (`results/*.json`) to GitHub because they are way too huge. Just run the notebooks yourself to generate them!

## Known limitations

- To keep things simple, the questions the AI asks are pulled from the differences in its own code, rather than having the AI write the questions completely from scratch.
- If the AI writes code that uses custom objects (without a basic string representation), the testing script sometimes gets confused and thinks the answers are different when they aren't.
- In the automated experiments, the "person" answering the questions is just the dataset's answer key. But the Part 4 notebook provides a real interactive mode where *you* can answer the questions yourself.