import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# settings
N_CANDIDATES = 8    # paper: 16
M_TESTS = 8         # paper: 16
K_MAX = 4           # max questions (paper: 4)
EPS = 0.02          # answer noise (paper: 0.02)
GAMMA = 0.85        # higher = asks more (paper: 0.85)
ETA = 0.5           
LAMBDA = 0.5
RHO = 0.5
MIN_COVERAGE = 0.5  # skip tests that crash on too many candidates

# True: prompt also shows the first test (standard MBPP style)
# keep it the same for all runs you compare
SHOW_EXAMPLE = True

# for 1.5B: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
print("Model loaded!")

SYSTEM_MSG = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# tokens used for the current problem (prompt + output)
tokens_used = 0


# reset token count
def reset_tokens():
    global tokens_used
    tokens_used = 0


# get token count
def get_tokens():
    return tokens_used


# send one prompt to the model, get n answers back (also counts tokens)
def run_model(user_text: str, n: int = 1, temperature: float = 0.8, max_new_tokens: int = 512):
    global tokens_used
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    if temperature == 0:
        # greedy, for one-shot
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    else:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            num_return_sequences=n,
            pad_token_id=tokenizer.eos_token_id,
        )

    texts = []
    for ids in output_ids:
        # decode only the new tokens, not the prompt
        text = tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
        texts.append(text)
        tokens_used += len(tokenizer.encode(text))
    tokens_used += prompt_len
    return texts


# name of the function we must write (the one used in the test)
def get_func_name(reference_code, test_line):
    names = re.findall(r"def\s+(\w+)\s*\(", reference_code)
    for name in names:
        if name + "(" in test_line:
            return name
    return names[0]


# "def name(args):" line from the reference code
def get_signature(reference_code, func_name):
    match = re.search(r"def\s+" + func_name + r"\s*\(.*?\)[^:\n]*:", reference_code, re.DOTALL)
    if match:
        return match.group(0)
    return "def " + func_name + "(...):"


# prompt for writing code (same for EIG and one-shot)
def build_prompt(problem_text: str, signature: str, example: str) -> str:
    prompt = (
        "Solve in Python.\n\n"
        f"Problem:\n{problem_text}\n\n"
        f"Use exactly this function signature:\n{signature}\n\n"
    )
    if SHOW_EXAMPLE:
        prompt += f"Your code must pass this test:\n{example}\n\n"
    prompt += (
        "If you need any class or helper function, define it yourself.\n\n"
        "IMPORTANT: Respond with ONLY a single Python code block."
    )
    return prompt


# get the code block out of the model answer
def extract_code(response_text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", response_text, re.DOTALL)
    if not match:
        match = re.search(r"```\s*(.*?)```", response_text, re.DOTALL)
    code = match.group(1) if match else response_text

    # remove extra print/assert/input lines, they can crash the code
    lines = []
    for line in code.splitlines():
        if line.startswith("print(") or line.startswith("assert ") or line.startswith("input("):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


# make N candidate programs
def generate_candidates(problem_text: str, signature: str, example: str, n: int = N_CANDIDATES):
    """Sample N different candidate programs"""
    prompt = build_prompt(problem_text, signature, example)

    # half the candidates try another meaning, for more variety
    other_prompt = prompt + (
        "\n\nNote: this problem can be understood in more than one way. "
        "Think about what else it could mean and write code for a different "
        "but still reasonable meaning."
    )
    half = n // 2
    texts = run_model(prompt, n=n - half, temperature=0.8)
    texts += run_model(other_prompt, n=half, temperature=1.0)
    return [extract_code(t) for t in texts]


# one-shot baseline: one greedy answer, no questions
def generate_one_shot(problem_text: str, signature: str, example: str):
    text = run_model(build_prompt(problem_text, signature, example), n=1, temperature=0)[0]
    return extract_code(text)


# ask the model for test asserts (our clarification questions)
def generate_candidate_tests(problem_text: str, signature: str, func_name: str, example: str, m: int = M_TESTS):
    instructions = (
        "You are helping disambiguate a Python coding problem.\n\n"
        f"Problem:\n{problem_text}\n\n"
        f"Function signature:\n{signature}\n\n"
    )
    if SHOW_EXAMPLE:
        instructions += f"Example test:\n{example}\n\n"
    instructions += (
        f"Propose {m} DIFFERENT test assertions of the form "
        f"`assert {func_name}(...) == ...` that would help find out what the "
        "problem really means. Try edge cases like empty inputs, negative numbers, "
        "duplicates or ties.\n\n"
        "Write ONLY the assert lines, one per line, with no numbering and no explanation."
    )
    text = run_model(instructions, n=1, temperature=0.9, max_new_tokens=400)[0]

    tests = []
    for line in text.splitlines():
        if "assert " not in line:
            continue
        # handle "1. assert ..." and backticks
        line = line[line.index("assert "):].strip().strip("`").strip()
        if func_name + "(" not in line:
            continue
        # skip broken lines
        try:
            compile(line, "<test>", "exec")
        except SyntaxError:
            continue
        # example is already in the prompt, no need to ask it
        if SHOW_EXAMPLE and line == example:
            continue
        if line not in tests:
            tests.append(line)
    return tests[:m]
