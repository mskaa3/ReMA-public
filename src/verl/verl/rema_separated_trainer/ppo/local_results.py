"""One conservative local-result boundary for communication and evaluation."""

import re
from typing import Optional


def extract_complete_boxed_answer(response: str) -> Optional[str]:
    """Return the last box; a malformed last box never falls back to an earlier one."""
    if not isinstance(response, str):
        return None
    matches = list(re.finditer(r"\\boxed\s*\{", response))
    if not matches:
        return None
    start = matches[-1].end() - 1
    depth = 0
    for index in range(start, len(response)):
        if response[index] == "{":
            depth += 1
        elif response[index] == "}":
            depth -= 1
            if depth == 0:
                return response[start + 1:index].strip() or None
    return None


def extract_local_result(output: str, max_chars: int = 600) -> str:
    """Prefer an explicit result, otherwise accept exactly one complete box.

    Do not guess from a prose sentence or choose between multiple unlabelled
    boxes. Never truncate a mathematical result into a different expression.
    """
    if not isinstance(output, str) or not output.strip():
        return ""
    declarations = list(re.finditer(r"local[_ ]result\s*:", output, re.IGNORECASE))
    if len(declarations) > 1:
        return ""
    if declarations:
        candidate = re.split(
            r"\n\s*(?:reasoning\s*:|subtask\b)",
            output[declarations[0].end():], maxsplit=1, flags=re.IGNORECASE,
        )[0].strip()
    else:
        candidate = output
    if len(re.findall(r"\\boxed\b", candidate)) != 1:
        return ""
    answer = extract_complete_boxed_answer(candidate)
    if answer is None:
        return ""
    # Without a declaration only the box, not the surrounding reasoning, is routed.
    result = candidate if declarations else "\\boxed{" + answer + "}"
    result = " ".join(result.split())
    return result if len(result) <= max_chars else ""
