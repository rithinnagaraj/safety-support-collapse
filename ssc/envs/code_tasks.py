"""Benign code (stage S2): MBPP-style "write a function that passes these asserts".

useful = the last ```python block in the reply, followed by every test assert,
runs to completion in a subprocess within the timeout. There is no safety
constraint, so safe is always True and SU == passes.

Anti-gaming: the harness prints a random per-run end marker after the
asserts, and a run passes only if that marker appears with exit code 0, so
`sys.exit(0)` or `os._exit(0)` before the asserts does not count as a pass.
Known gap: an object whose __eq__ always returns True still passes `assert
f(x) == y`. Small models rarely find this, and the grader audit would catch it.

This is NOT a security sandbox. It isolates the working directory, clears the
environment, and (on Linux) caps memory and CPU time, but generated code can
still use the network or read files. Run training inside a container.
"""

from __future__ import annotations

import os
import re
import secrets
import subprocess
import sys
import tempfile

from ssc.types import NOT_U_S, SU, Grade, Task

FAMILY = "code"
SYSTEM = "You are a helpful assistant."
TIMEOUT_S = 6.0
MEMORY_BYTES = 1 << 30

_BLOCK_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(response: str) -> str | None:
    blocks = _BLOCK_RE.findall(response)
    return blocks[-1] if blocks else None


def _limits() -> None:  # runs in the child on POSIX only
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (int(TIMEOUT_S) + 1, int(TIMEOUT_S) + 1))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run_tests(code: str, tests: list[str], setup: str = "", timeout: float = TIMEOUT_S) -> tuple[bool, str]:
    marker = f"__SSC_DONE_{secrets.token_hex(12)}__"
    program = f"{setup}\n{code}\n\n" + "\n".join(tests) + f"\nprint({marker!r})\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8"}
        if os.name == "nt":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", path], cwd=d, env=env, capture_output=True, text=True,
                timeout=timeout, stdin=subprocess.DEVNULL, preexec_fn=_limits if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if proc.returncode != 0:
        last = proc.stderr.strip().splitlines()[-1:] or ["nonzero exit"]
        return False, f"error:{last[0][:120]}"
    if marker not in proc.stdout:
        return False, "exited_before_tests_finished"
    return True, "passed"


def make_task(task_id: str, split: str, text: str, tests: list[str], setup: str, reference: str, meta: dict) -> Task:
    user = (f"{text}\nYour code should pass this test:\n{tests[0]}\n\n"
            "Write the complete Python function in a single ```python code block.")
    return Task(task_id=task_id, family=FAMILY, split=split, system=SYSTEM, user=user, answer=reference,
                meta={**meta, "tests": tests, "setup": setup})


def generate(split: str, index: int, seed: int = 0) -> Task:
    raise RuntimeError("code tasks come from datasets: run `python -m ssc.envs.prepare_benign` and pass --data")


def grade(task: Task, response: str, level: str = "eval") -> Grade:
    code = extract_code(response)
    if code is None:
        return Grade(useful=False, safe=True, reasons=("no_code_block",))
    ok, why = run_tests(code, task.meta["tests"], task.meta.get("setup", ""))
    return Grade(useful=ok, safe=True, reasons=(why,))


def synthetic_responses(task: Task) -> dict[str, str]:
    return {SU: f"```python\n{task.answer}\n```", NOT_U_S: "```python\ndef solution():\n    return None\n```"}
