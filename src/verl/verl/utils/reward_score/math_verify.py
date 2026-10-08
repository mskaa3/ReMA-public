# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import lru_cache
import re


@lru_cache(maxsize=1)
def _verifier():
    try:
        from math_verify import parse, verify
        from math_verify.parser import LatexExtractionConfig
    except ImportError as exc:
        raise RuntimeError(
            "Mathematical answer verification requires math-verify==0.7.0; "
            "install the project requirements in the training container. "
            "Refusing to silently replace mathematical verification with string matching."
        ) from exc
    return parse, verify, LatexExtractionConfig


def _normalize_fraction_grouping(text: str) -> str:
    """Remove thousands separators only in complete numeric fraction arguments."""
    replacements = []
    for match in re.finditer(r"\\(?:frac|dfrac|tfrac)\b", text):
        cursor = match.end()
        for _ in range(2):
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor >= len(text) or text[cursor] != "{":
                break
            start = cursor + 1
            depth = 1
            cursor += 1
            while cursor < len(text) and depth:
                if text[cursor] == "{":
                    depth += 1
                elif text[cursor] == "}":
                    depth -= 1
                cursor += 1
            if depth:
                break
            end = cursor - 1
            argument = text[start:end].strip()
            argument = re.sub(r"\\(?:[!,;:]|(?:thinspace|negthinspace)\b)\s*", "", argument)
            if re.fullmatch(r"[+-]?\d{1,3}(?:,\s*\d{3})+(?:\.\d+)?", argument):
                replacements.append((start, end, re.sub(r"[,\s]", "", argument)))
    # Never remove commas globally: they can delimit coordinates, sets or answers.
    for start, end, value in sorted(replacements, reverse=True):
        text = text[:start] + value + text[end:]
    return text


def _normalize_notation(text: str) -> str:
    # Evaluation-only normalization: never modify downstream/model artifacts.
    text = str(text or "").strip().translate(str.maketrans({
        "\u03c0": r"\pi ", "\u221e": r"\infty ", "\u2212": "-",
        "\u00d7": r"\times ", "\u00b7": r"\cdot ", "\u00f7": "/",
        "\u2264": r"\leq ", "\u2265": r"\geq ", "\u221a": r"\sqrt ",
    }))
    text = re.sub(r"\\(?:left|right)\b", "", text)
    while len(text) >= 2 and text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()
    for opening, closing in ((r"\(", r"\)"), (r"\[", r"\]")):
        if text.startswith(opening) and text.endswith(closing):
            text = text[len(opening):-len(closing)].strip()
    text = _normalize_fraction_grouping(text)
    # Parse standalone percentages as exact fractions. LaTeX parsers can reject
    # the percent suffix or silently strip it when wrapped in \text{...}.
    percentage = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
        r"(?:\\?%|\\(?:text|mathrm)\{\s*\\?%\s*\})",
        text,
    )
    if percentage:
        return r"\frac{" + percentage.group(1) + "}{100}"
    return text


def compute_answer_score(model_output: str, ground_truth: str) -> float:
    """Compare complete answer expressions, not a matching number within one."""
    prediction = _normalize_notation(model_output)
    reference = _normalize_notation(ground_truth)
    if not prediction or not reference:
        return 0.0
    if prediction == reference:
        return 1.0
    # Do not interpret multiple-choice letters as algebraic variables.
    ref_choice = re.fullmatch(r"(?:\\text\{)?([A-E])\}?", reference)
    if ref_choice:
        pred_choice = re.fullmatch(r"(?:\\text\{)?([A-E])\}?", prediction)
        return float(bool(pred_choice and pred_choice[1] == ref_choice[1]))

    parse, verify, latex_config = _verifier()
    kwargs = dict(extraction_config=[latex_config()], fallback_mode="no_fallback",
                  extraction_mode="first_match")
    # The caller supplies the extracted terminal answer. Wrapping the whole
    # expression avoids dropping pi, intervals, fractions or plain sqrt(...).
    gold = parse("$" + reference + "$", **kwargs)
    answer = parse("$" + prediction + "$", **kwargs)
    if not gold:
        raise ValueError(f"Cannot parse ground-truth mathematical answer: {ground_truth!r}")
    return float(bool(answer) and verify(gold, answer))


def compute_score(model_output: str, ground_truth: str) -> float:
    """Keep full-solution extraction for callers outside hierarchical ReMA."""
    _verifier()  # Report missing dependencies instead of silently returning zero.
    from math_verify.metric import math_metric
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
    metric = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    score, _ = metric([r"\boxed{" + ground_truth + "}"], [model_output])
    return float(score)
