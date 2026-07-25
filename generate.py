import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Hyperparameters 
N_CANDIDATES = 8   # paper uses 16 
M_TESTS = 8         # paper uses 16
K_MAX = 4           # (paper: K=4)
EPS = 0.02          # flip probability (paper: 0.02)
GAMMA = 0.85        # (paper: 0.85)
ETA = 0.5           
LAMBDA = 0.5       
RHO = 0.5           

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
print("Model loaded!")


def build_prompt(problem_text: str, test_example: str) -> str:
    instructions = (
        "Solve in Python.\n\n"
        f"Problem:\n{problem_text}\n\n"
        "Your code must satisfy this test case exactly "
        "(same function name, same argument order):\n"
        f"{test_example}\n\n"
        "If the test uses any class or object that isn't a Python "
        "builtin, you must define that class yourself as part of "
        "your code.\n\n"
        "IMPORTANT: Respond with ONLY a single Python code block."
    )
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": instructions},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
def extract_code(response_text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()


def generate_candidates(problem_text: str, test_example: str, n: int = N_CANDIDATES, max_new_tokens: int = 512):
    """Sample N different candidate programs"""
    prompt = build_prompt(problem_text, test_example)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        num_return_sequences=n,
        pad_token_id=tokenizer.eos_token_id,
    )

    return [extract_code(tokenizer.decode(ids, skip_special_tokens=True)) for ids in output_ids]


def generate_candidate_tests(problem_text: str, test_example: str, m: int = M_TESTS, max_new_tokens: int = 400):
    func_hint = test_example.split("(")[0].replace("assert", "").strip()

    instructions = (
        "You are helping disambiguate a Python coding problem.\n\n"
        f"Problem:\n{problem_text}\n\n"
        f"The function under test is called `{func_hint}` "
        f"(see example: {test_example}).\n\n"
        f"Propose {m} DIFFERENT test assertions of the form "
        f"`assert {func_hint}(...) == ...` that would help distinguish "
    )
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": instructions},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    lines = [l.strip() for l in text.splitlines()]
    tests = [l for l in lines if l.startswith("assert ") and func_hint in l]
    return list(dict.fromkeys(tests))[:m] 
