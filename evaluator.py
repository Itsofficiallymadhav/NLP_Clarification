import contextlib
import io
import multiprocessing

# each test runs in a new process with a time limit, so an infinite loop can't block the run
# "fork" means the model is not loaded again in the new process


# runs in the new process: runs the code and one test, sends back the result
def _run_single_test(code_str, test_line, setup_code, queue):
    try:
        namespace = {}
        with contextlib.redirect_stdout(io.StringIO()):
            if setup_code:
                exec(setup_code, namespace)
            exec(code_str, namespace)
            try:
                exec(test_line, namespace)
                queue.put(1)          # passed
            except AssertionError:
                queue.put(0)          # cleanly failed
    except Exception:
        queue.put(None)              # undefined / crashed


# run one test on one piece of code
def run_test(code_str, test_line, setup_code="", timeout=5):
    """Returns 1 (pass), 0 (fail), or None (undefined / crashed)."""
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    process = ctx.Process(target=_run_single_test, args=(code_str, test_line, setup_code, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return None
    try:
        return queue.get(timeout=1)
    except Exception:
        # no result, e.g. the code called exit()
        return None


# oracle: answer a question using the reference code
def oracle_answer(reference_code, test_line, setup_code=""):
    """The oracle: run a test against the REFERENCE implementation."""
    result = run_test(reference_code, test_line, setup_code)
    return result if result in (0, 1) else None


# like _run_single_test, but runs all hidden tests
def _run_full_eval(code_str, test_code, setup_code, queue):
    try:
        namespace = {}
        with contextlib.redirect_stdout(io.StringIO()):
            if setup_code:
                exec(setup_code, namespace)
            exec(code_str, namespace)
            exec(test_code, namespace)
        queue.put((True, None))
    except Exception as e:
        queue.put((False, f"{type(e).__name__}: {str(e)[:200]}"))


# check final code on hidden tests (for pass@1)
def evaluate_final(code_str, test_code, setup_code="", timeout=20):
    # test_code: MBPP asserts joined, or the MBPP+ test script
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    process = ctx.Process(target=_run_full_eval, args=(code_str, test_code, setup_code, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return False, f"Timed out after {timeout}s"
    try:
        return queue.get(timeout=1)
    except Exception:
        return False, "No result"
