import re
import ast

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