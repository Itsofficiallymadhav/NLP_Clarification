import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from evaluation.evaluator import extract_code
import ast

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
).to("cuda")

print("Model loaded!")

def build_prompt(problem, test_example=None):
    test_section = ""
    if test_example:
        test_section = f"""
Your code must satisfy this test case exactly (same function name, same argument order):
{test_example}

If the test uses any class or object (e.g. Pair) that isn't a Python builtin, you must define that class yourself as part of your code.
"""
    return f"""Solve the following problem in Python.

Problem:
{problem}
{test_section}

IMPORTANT: Respond with ONLY a single Python code block. Do not include any explanation, reasoning, or text before or after the code block."""


def generate(prompt, test_example=None, temperature=0.2, do_sample=False, top_p=0.95):
    prompt = build_prompt(prompt, test_example)
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    gen_kwargs = {
        "max_new_tokens": 512,
    }

    if do_sample:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        })
    else:
        gen_kwargs["do_sample"] = False  # greedy, deterministic

    outputs = model.generate(**inputs, **gen_kwargs)

    generated = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return generated


def filter_invalid(candidates):
    """Remove candidates that don't parse as valid Python."""
    valid = []
    for code in candidates:
        try:
            ast.parse(code)
            valid.append(code)
        except SyntaxError:
            continue
    return valid