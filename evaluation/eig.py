import math

def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def init_weights(candidates):
    n = len(candidates)
    return {i: 1.0 / n for i in range(n)}


def compute_pt(candidates, weights, run_fn, func_name, test_input, hypothesized_output):
    outcomes = {}
    for j, code in enumerate(candidates):
        result = run_fn(code, func_name, test_input)
        if result.startswith("ERROR"):
            outcomes[j] = None
        else:
            outcomes[j] = 1 if result == hypothesized_output else 0

    defined_weight = sum(weights[j] for j in outcomes if outcomes[j] is not None)
    if defined_weight == 0:
        return 0.5, outcomes, 0.0

    pass_weight = sum(weights[j] for j in outcomes if outcomes[j] == 1)
    pt = pass_weight / defined_weight
    return pt, outcomes, defined_weight


def eig_score(pt, epsilon=0.02):
    return binary_entropy(epsilon + (1 - 2 * epsilon) * pt) - binary_entropy(epsilon)


def full_score(pt, defined_weight, undefined_weight, epsilon=0.02, lam=0.5, rho=0.5):
    eig = eig_score(pt, epsilon)
    balance_bonus = (1 - lam) + lam * 4 * pt * (1 - pt)
    penalty = 1 - rho * undefined_weight
    return eig * defined_weight * balance_bonus * penalty


def build_test_pool(disagreement_report):
    test_pool = []
    for d in disagreement_report:
        inp = d["input"]
        for output_repr in d["output_groups"]:
            if output_repr.startswith("ERROR"):
                continue
            test_pool.append((tuple(inp), output_repr))
    return test_pool


def oracle_answer(reference_code, func_name, test_input, hypothesized_output_repr, run_fn):
    actual = run_fn(reference_code, func_name, test_input)
    return 1 if actual == hypothesized_output_repr else 0


def expected_pnext(candidates, weights, run_fn, func_name, test_input, hypothesized_output, epsilon=0.02, eta=0.3):
    pt, outcomes, _ = compute_pt(candidates, weights, run_fn, func_name, test_input, hypothesized_output)

    def simulate_update(answer):
        new_weights = {}
        for j, w in weights.items():
            outcome = outcomes.get(j)
            if outcome is None:
                factor = eta
            elif outcome == answer:
                factor = 1 - epsilon
            else:
                factor = epsilon
            new_weights[j] = w * factor
        total = sum(new_weights.values())
        if total == 0:
            return max(weights.values())
        return max(v / total for v in new_weights.values())

    p_answer_1 = epsilon + (1 - 2 * epsilon) * pt
    p_answer_0 = 1 - p_answer_1
    return p_answer_1 * simulate_update(1) + p_answer_0 * simulate_update(0)


def should_ask(candidates, weights, run_fn, func_name, test_input, hypothesized_output, gamma=0.85):
    pnow = max(weights.values())
    pnext = expected_pnext(candidates, weights, run_fn, func_name, test_input, hypothesized_output)
    return gamma * pnext > pnow


def update_weights(weights, outcomes, oracle_answer_val, epsilon=0.02, eta=0.3):
    new_weights = {}
    for j, w in weights.items():
        outcome = outcomes.get(j)
        if outcome is None:
            factor = eta
        elif outcome == oracle_answer_val:
            factor = 1 - epsilon
        else:
            factor = epsilon
        new_weights[j] = w * factor

    total = sum(new_weights.values())
    if total == 0:
        return weights
    return {j: v / total for j, v in new_weights.items()}