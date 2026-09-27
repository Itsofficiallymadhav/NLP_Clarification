import re

from generate import (N_CANDIDATES, M_TESTS, K_MAX, SHOW_EXAMPLE, generate_candidates,
                      generate_candidate_tests, generate_one_shot, get_func_name, get_signature,
                      reset_tokens, get_tokens)
from scoring import score_test, should_ask, update_weights
from evaluator import oracle_answer, run_test


# remove bad candidates and merge duplicates (duplicates get more weight)
def clean_candidates(candidates, func_name, example, setup_code):
    counts = {}
    for c in candidates:
        # must define the needed function
        if not re.search(r"def\s+" + func_name + r"\s*\(", c):
            continue
        # must be valid python
        try:
            compile(c, "<candidate>", "exec")
        except SyntaxError:
            continue
        counts[c] = counts.get(c, 0) + 1

    # the example test is visible, so failing it means the candidate is wrong
    if SHOW_EXAMPLE:
        passing = {}
        for c in counts:
            if run_test(c, example, setup_code) == 1:
                passing[c] = counts[c]
        if len(passing) > 0:
            counts = passing

    if len(counts) == 0:
        # nothing valid, keep all with equal weight
        return candidates, [1.0 / len(candidates)] * len(candidates)

    unique = list(counts.keys())
    total = sum(counts.values())
    weights = [counts[c] / total for c in unique]
    return unique, weights


# main EIG loop: candidates -> ask up to K questions -> return best candidate
def run_clarification_pipeline(problem, verbose=False):
    reset_tokens()
    problem_text = problem["text"]
    reference_code = problem["code"]
    setup_code = problem["setup"]
    example = problem["example"]
    func_name = get_func_name(reference_code, example)
    signature = get_signature(reference_code, func_name)

    raw_candidates = generate_candidates(problem_text, signature, example, n=N_CANDIDATES)
    candidates, weights = clean_candidates(raw_candidates, func_name, example, setup_code)
    if verbose:
        print(f"{len(raw_candidates)} candidates -> {len(candidates)} after cleaning")

    questions_asked = 0
    asked = []   # don't ask the same test twice
    log = []

    for round_num in range(K_MAX):
        # one candidate left, nothing to ask
        if len(candidates) == 1:
            break

        raw_tests = generate_candidate_tests(problem_text, signature, func_name, example, m=M_TESTS)

        scored = []
        for t in raw_tests:
            if t in asked:
                continue
            result = score_test(t, candidates, weights, setup_code)
            if result is not None:
                scored.append(result)
        if len(scored) == 0:
            continue   # no useful test, try next round

        best = max(scored, key=lambda r: r["score"])
        if not should_ask(best, weights):
            if verbose:
                print(f"Round {round_num}: not worth asking further (p_now high enough)")
            break

        asked.append(best["test"])
        oracle_result = oracle_answer(reference_code, best["test"], setup_code)
        if oracle_result is None:
            continue   # test crashed on the reference, skip it
        weights = update_weights(weights, best["outcomes"], oracle_result)
        questions_asked += 1

        log.append({
            "round": round_num, "test": best["test"], "score": best["score"],
            "oracle_answer": oracle_result, "weights_after": weights,
        })

        if verbose:
            print(f"Round {round_num}: asked `{best['test']}` -> oracle={oracle_result}")
            print(f"  weights: {[round(w, 3) for w in weights]}")

    best_idx = weights.index(max(weights))
    return {
        "final_code": candidates[best_idx],
        "weights": weights,
        "candidates": candidates,
        "questions_asked": questions_asked,
        "tokens": get_tokens(),
        "log": log,
    }


# one-shot baseline: same prompt, one answer, no questions
def run_one_shot(problem):
    reset_tokens()
    func_name = get_func_name(problem["code"], problem["example"])
    signature = get_signature(problem["code"], func_name)
    code = generate_one_shot(problem["text"], signature, problem["example"])
    return {"final_code": code, "questions_asked": 0, "tokens": get_tokens()}
