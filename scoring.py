import math

from generate import EPS, GAMMA, ETA, LAMBDA, RHO
from evaluator import run_test


def binary_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def score_test(test_line, candidates, weights, reference_code, setup_code=""):
    """Computes f_t(c_j) for every candidate, then p_t, EIG(t), and
    the full score(t)"""
    outcomes = [run_test(c, test_line, setup_code) for c in candidates]

    defined_mask = [o is not None for o in outcomes]
    w_defined_total = sum(w for w, d in zip(weights, defined_mask) if d)
    w_undefined_total = sum(w for w, d in zip(weights, defined_mask) if not d)

    if w_defined_total == 0:
        return None  

    w_pass = sum(w for w, o in zip(weights, outcomes) if o == 1)
    p_t = w_pass / w_defined_total

    eig = binary_entropy(EPS + (1 - 2 * EPS) * p_t) - binary_entropy(EPS)

    balance_bonus = (1 - LAMBDA) + LAMBDA * 4 * p_t * (1 - p_t)
    crash_penalty = (1 - RHO * w_undefined_total)
    score = eig * w_defined_total * balance_bonus * crash_penalty

    return {
        "test": test_line, "outcomes": outcomes, "p_t": p_t,
        "eig": eig, "score": score,
        "w_def": w_defined_total, "u_t": w_undefined_total,
    }


def update_weights(weights, outcomes, oracle_result):
    """Soft Bayesian update."""
    new_weights = []
    for w, o in zip(weights, outcomes):
        if o is None:
            factor = ETA
        elif o == oracle_result:
            factor = (1 - EPS)
        else:
            factor = EPS
        new_weights.append(w * factor)

    total = sum(new_weights)
    if total == 0:
        return weights 
    return [w / total for w in new_weights]


def expected_p_next(test_result, weights):
    """Approximates p_next(t) ."""
    outcomes = test_result["outcomes"]
    p_t = test_result["p_t"]

    w_if_pass = update_weights(weights, outcomes, oracle_result=1)
    w_if_fail = update_weights(weights, outcomes, oracle_result=0)

    return p_t * max(w_if_pass) + (1 - p_t) * max(w_if_fail)


def should_ask(test_result, weights):
    """ask if gamma * p_next(t) > p_now"""
    p_now = max(weights)
    p_next = expected_p_next(test_result, weights)
    return (GAMMA * p_next) > p_now
