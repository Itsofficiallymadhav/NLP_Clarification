import math

from generate import EPS, GAMMA, ETA, LAMBDA, RHO, MIN_COVERAGE
from evaluator import run_test


# binary entropy H(p)
def binary_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


# score one test question (paper, sections 4.2-4.3), higher is better
def score_test(test_line, candidates, weights, setup_code=""):
    """Computes f_t(c_j) for every candidate, then p_t, EIG(t), and
    the full score(t)"""
    outcomes = [run_test(c, test_line, setup_code) for c in candidates]

    defined_mask = [o is not None for o in outcomes]
    w_defined_total = sum(w for w, d in zip(weights, defined_mask) if d)
    w_undefined_total = sum(w for w, d in zip(weights, defined_mask) if not d)

    # skip tests that crash on too many candidates (paper, 5.2)
    if w_defined_total < MIN_COVERAGE:
        return None
    # skip tests where all candidates agree
    defined_outcomes = [o for o in outcomes if o is not None]
    if all(o == defined_outcomes[0] for o in defined_outcomes):
        return None

    # p_t: weighted chance the test passes
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


# update weights after the oracle answers
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


# expected top weight after asking this test
def expected_p_next(test_result, weights):
    """Approximates p_next(t) ."""
    outcomes = test_result["outcomes"]
    # chance the oracle says pass (with noise)
    p_yes = EPS + (1 - 2 * EPS) * test_result["p_t"]

    w_if_pass = update_weights(weights, outcomes, oracle_result=1)
    w_if_fail = update_weights(weights, outcomes, oracle_result=0)

    return p_yes * max(w_if_pass) + (1 - p_yes) * max(w_if_fail)


# ask another question or submit?
def should_ask(test_result, weights):
    """ask if gamma * p_next(t) > p_now"""
    p_now = max(weights)
    p_next = expected_p_next(test_result, weights)
    return (GAMMA * p_next) > p_now
