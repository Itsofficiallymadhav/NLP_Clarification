import ast
import hashlib
import re

def extract_code(response):
    if not response:
        return ""
    match = re.search(r"```python(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```python(.*)", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return response.strip()

def get_defined_functions(code):
    """Return list of top-level function names defined in code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]

def get_called_functions(tests):
    """Return set of function names called at the top level of each assert (not nested args)."""
    called = set()
    for test in tests:
        try:
            tree = ast.parse(test)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                expr = node.test
                if isinstance(expr, ast.Compare) and isinstance(expr.left, ast.Call):
                    if isinstance(expr.left.func, ast.Name):
                        called.add(expr.left.func.id)
    return called

def evaluate(response, tests, setup_code=""):
    code = extract_code(response)
    program = ""
    if setup_code:
        program += extract_code(setup_code) + "\n\n"
    program += code + "\n\n"
    defined = get_defined_functions(code)
    expected = get_called_functions(tests)
    missing = expected - set(defined)
    if len(missing) == 1 and len(defined) == 1:
        alias_name = missing.pop()
        program += f"{alias_name} = {defined[0]}\n\n"
    elif missing and defined:
        for alias_name in missing:
            program += f"{alias_name} = {defined[-1]}\n\n"
    for test in tests:
        program += test + "\n"
    try:
        namespace = {}
        exec(program, namespace)
        return True, None
    except AssertionError as e:
        return False, f"AssertionError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    

def get_function_name(code):
    """Extract the name of the first top-level function defined in code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None


def run_candidate(code, func_name, args):
    """Execute candidate code and call func_name(*args). Returns a comparable, normalized repr."""
    namespace = {}
    try:
        exec(code, namespace)
        func = namespace.get(func_name)
        if func is None:
            return "ERROR: function not found"
        result = func(*args)

        # Normalize generators/maps into concrete lists so repr is stable
        if hasattr(result, '__iter__') and not isinstance(result, (str, list, tuple, dict, set)):
            try:
                result = list(result)
            except Exception:
                pass

        return repr(result)
    except Exception as e:
        return f"ERROR: {type(e).__name__}"


def deduplicate_candidates(candidates, probe_inputs, expected_func_name):
    """
    Deduplicate candidates by behavior on probe_inputs.
    probe_inputs: list of arg-tuples to test each candidate with.
    expected_func_name: the function name the tests expect (from test_list),
                         used to alias mismatched names before running.
    """
    seen_signatures = {}
    unique_candidates = []

    for code in candidates:
        func_name = get_function_name(code)
        if func_name is None:
            continue

        # Alias to expected name if candidate used a different name
        exec_code = code
        if func_name != expected_func_name:
            exec_code += f"\n\n{expected_func_name} = {func_name}\n"

        # Build a "behavior signature": tuple of outputs on all probe inputs
        outputs = tuple(
            run_candidate(exec_code, expected_func_name, args)
            for args in probe_inputs
        )
        signature = hashlib.md5(str(outputs).encode()).hexdigest()

        if signature not in seen_signatures:
            seen_signatures[signature] = code
            unique_candidates.append(code)

    return unique_candidates


def extract_call_args(test_str):
    """
    Given a test string like 'assert f(1, 2) == 3', extract the argument tuple (1, 2).
    Returns None if parsing fails.
    """
    try:
        tree = ast.parse(test_str)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            expr = node.test
            if isinstance(expr, ast.Compare) and isinstance(expr.left, ast.Call):
                call = expr.left
                try:
                    # Evaluate each argument node as a literal (safe, no code execution)
                    args = tuple(ast.literal_eval(arg) for arg in call.args)
                    return args
                except (ValueError, SyntaxError):
                    return None  # arg isn't a simple literal (e.g. Pair(...) object)
    return None


def get_probe_inputs(test_list):
    """Extract probe input tuples from all tests in test_list."""
    probes = []
    for test in test_list:
        args = extract_call_args(test)
        if args is not None:
            probes.append(args)
    return probes


def find_disagreements(candidates, probe_inputs, expected_func_name):
    """For each probe input, check whether candidates disagree on output.
    Only counts disagreement among candidates that ran without crashing."""
    disagreement_report = []

    for inp in probe_inputs:
        output_groups = {}

        for idx, code in enumerate(candidates):
            func_name = get_function_name(code)
            if func_name is None:
                continue

            exec_code = code
            if func_name != expected_func_name:
                exec_code += f"\n\n{expected_func_name} = {func_name}\n"

            result = run_candidate(exec_code, expected_func_name, inp)

            if result.startswith("ERROR:"):
                continue  # skip crashed candidates entirely

            output_groups.setdefault(result, []).append(idx)

        if len(output_groups) > 1:
            disagreement_report.append({
                "input": inp,
                "output_groups": output_groups
            })

    return disagreement_report