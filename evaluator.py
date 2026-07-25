import contextlib
import io
import multiprocessing


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
    return queue.get() if not queue.empty() else None


def oracle_answer(reference_code, test_line, setup_code=""):
    """The oracle: run a test against the REFERENCE implementation."""
    result = run_test(reference_code, test_line, setup_code)
    return result if result in (0, 1) else None


def _run_full_eval(code_str, test_list, setup_code, queue):
    try:
        namespace = {}
        with contextlib.redirect_stdout(io.StringIO()):
            if setup_code:
                exec(setup_code, namespace)
            exec(code_str, namespace)
            for test in test_list:
                exec(test, namespace)
        queue.put((True, None))
    except Exception as e:
        queue.put((False, f"{type(e).__name__}: {e}"))


def evaluate_final(code_str, test_list, setup_code="", timeout=10):
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    process = ctx.Process(target=_run_full_eval, args=(code_str, test_list, setup_code, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return False, f"Timed out after {timeout}s"
    return queue.get() if not queue.empty() else (False, "No result")
