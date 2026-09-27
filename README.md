# NLP Clarification: EIG with generated test questions

In general, you sometimes face problems with the outputs given by an LLM, mostly because the prompt you gave is ambiguous and the model has to guess. Bigger LLMs usually solve this by using a lot of internal thinking, which costs too many tokens. This is what I am trying to solve. I use a small LLM and help it give better answers without using that many tokens, by asking clarification questions about the code, which are answered by an oracle (the reference solution). My project asks one or more questions (only if needed) to choose the best solution from several generated candidates.

> Branch `eig-generated-tests` of the NLP Clarification project.
> The other approach is on branch `eig-7b-disagreement`.

## Based on

"20 Questions for Code: Improving Code Generation with Information-Theoretic Clarification"
by Ria Garg, Alexandra Kim and Julia Xi (Stanford CS224N).

My EIG-based approach takes several ideas from the paper mentioned above, mainly the math for scoring questions and the gamma value used to decide whether to ask another question or stop. I reduced the number of candidate programs from 16 to 8, and the number of proposed tests per round from 16 to 8, because of limited free GPU time. I also ran it on the MBPP+ dataset, which has many more hidden tests per problem, to check the results more strictly.

## How it works

1. **Candidates:** These are the sample solutions generated immediately after getting the prompt.
2. **Test questions:** These are questions, in the form of unit tests, that the LLM generates to find the best code among all the candidates.
3. **EIG score:** This is the expected reduction in uncertainty (entropy) about the candidates after asking a test question.
4. **Oracle and weight update:** The oracle is the reference solution from the dataset. It answers each test question with yes or no. All candidates start with equal weights, and after each oracle answer, candidates that agree with it are rewarded and the others are punished.
5. **Ask or submit:** Sometimes one test question is enough to clear up all the ambiguity, so we need to decide whether to ask another question or stop. We ask again only if `gamma * p_next > p_now`.

## How this branch differs from `eig-7b-disagreement`

In my approach, the LLM invents new test questions itself, while in the `eig-7b-disagreement` approach, the questions are built from the inputs of the hidden tests that are also used for the final score. I also used the MBPP+ dataset, which has many more hidden tests per problem, to check my results more strictly.

## Results

First 25 problems of each dataset (MBPP sanitized test split, MBPP+).

| Model | Dataset | Method | pass@1 | Avg. questions | Avg. tokens |
|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | MBPP | One-shot | 0.68 | 0.00 | 171 |
| Qwen2.5-Coder-1.5B | MBPP | EIG | 0.80 | 0.60 | 2628 |
| Qwen2.5-Coder-3B | MBPP | EIG | 0.84 | 0.56 | 3076 |
| Qwen2.5-Coder-3B | MBPP+ | EIG | 0.88 | 0.56 | 2994 |

Note: the first 25 problems of MBPP+ are not the same as the first 25 of MBPP. Some easy problems are only in the MBPP+ list, which is why its score is higher even though its tests are stricter.

Comparing the pass@1 scores suggests that asking questions improves the chances of getting the right answer: one-shot, which asks no questions, scored 0.68, while the EIG-based approach scored 0.80 on the same 25 problems. EIG used about 15 times more tokens, but the original paper showed that a small model with EIG can come close to a much larger model while still using fewer tokens.

## How to run

```bash
pip install -r requirements.txt
```

1. Open `eig_pipeline.ipynb`.
2. Set `PROJECT_DIR` in the Setup cell to the folder with the `.py` files.
3. Pick the model with `MODEL_NAME` in `generate.py` (1.5B or 3B), then restart the kernel.
4. Run the cells from the top. Results are saved to `results/`.

You can run the code on a free Google Colab T4 GPU, which is what I used. It should also work on Kaggle with the T4 GPU accelerator.

## Files

| File / folder | What it is |
|---|---|
| `generate.py` | Loads the model, makes the prompts, and generates the candidate codes and test questions. |
| `evaluator.py` | Runs the code and tests safely in a separate process with a time limit. Also has the oracle. |
| `scoring.py` | Calculates the EIG score of each test question, updates the weights, and decides whether to ask or stop. |
| `clarification.py` | The main EIG loop that connects everything, plus the one-shot baseline. |
| `eig_pipeline.ipynb` | The notebook to run all the experiments and see the results. |
| `results/` | The saved results of every run, one line per problem, with the settings used. |
| `report/` | The project report (LaTeX file and PDF). |

## Limitations

- The experiments currently run on only 25 problems, which is not enough to give a reliable pass@1 score. In the future, I plan to run them on 50 and then 100 problems.
- The models I used are small Qwen models (1.5B and 3B). A 7B model would likely write better candidate codes.
- The model currently generates only 8 candidate codes per problem, while the paper uses 16. More candidates give a better chance that at least one of them is correct, so EIG has a better option to choose from. All of these limits come from the limited free GPU time.
