"""Evaluation-only stated-answer extraction; never used for routing or rewards."""

import re

from local_results import extract_complete_boxed_answer

VERSION = 'standalone-local-result-v1'


def reported_answer(response):
    """Ignore inline quotations, but never choose among competing result sections.

    This does not establish sound reasoning or role compliance. Without an
    explicit standalone declaration, require a single box in the whole output.
    In particular, do not fall back to an earlier box after a malformed result.
    """
    if not isinstance(response, str) or not response.strip():
        return dict(answer='', reason='empty')
    declarations = list(re.finditer(
        r'^[ \t]*(?:\*\*)?LOCAL_RESULT[ \t]*:(?:\*\*)?[ \t]*',
        response, re.MULTILINE))
    if len(declarations) > 1:
        return dict(answer='', reason='multiple_standalone_results')
    candidate = response[declarations[0].end():] if declarations else response
    boxes = re.findall(r'\\boxed\b', candidate)
    if len(boxes) != 1:
        return dict(answer='', reason='missing_or_ambiguous_box')
    answer = extract_complete_boxed_answer(candidate)
    if answer is None:
        return dict(answer='', reason='malformed_box')
    if len(answer) > 600:
        return dict(answer='', reason='oversized_answer')
    return dict(answer=answer, reason='standalone_result' if declarations else 'unique_box')
