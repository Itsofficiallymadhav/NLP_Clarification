from generate import N_CANDIDATES, M_TESTS, K_MAX, generate_candidates, generate_candidate_tests
from scoring import score_test, should_ask, update_weights
from evaluator import oracle_answer


def run_clarification_pipeline(sample, verbose=False):
    problem_text = sample["text"]
    test_example = sample["test_list"][0]
    reference_code = sample["code"]
    setup_code = sample.get("test_setup_code", "")

    candidates = generate_candidates(problem_text, test_example, n=N_CANDIDATES)
    weights = [1.0 / len(candidates)] * len(candidates)

    questions_asked = 0
    log = []

    for round_num in range(K_MAX):
        raw_tests = generate_candidate_tests(problem_text, test_example, m=M_TESTS)
        if not raw_tests:
            break
        scored = [r for t in raw_tests if (r := score_test(t, candidates, weights, reference_code, setup_code)) is not None]
        if not scored:
            break 
        best = max(scored, key=lambda r: r["score"])
        if not should_ask(best, weights):
            if verbose:
                print(f"Round {round_num}: not worth asking further (p_now high enough)")
            break

        oracle_result = oracle_answer(reference_code, best["test"], setup_code)
        if oracle_result is None:
            continue 
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
        "log": log,
    }
