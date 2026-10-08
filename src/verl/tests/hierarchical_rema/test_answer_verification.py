"""Math equivalence and terminal-answer regression tests (no model/GPU needed)."""
import importlib
from types import SimpleNamespace

import pytest

try:
    from verl.hierarchical_rema import rewarding
except ModuleNotFoundError:
    from hierarchical_rema import rewarding


@pytest.mark.parametrize("prediction,reference", [
    ("0.5", r"\frac{1}{2}"),
    ("686\u03c0", r"686\pi"),
    ("[6, \u221e)", r"[6,\infty)"),
    ("sqrt(13^2 - 2*24)", "11"),
    (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
    (r"\boxed{\frac{\sqrt{4}}{4}}", "0.5"),
    ("Therefore, the positive difference between the solutions is $\\frac{31}{6}$.", r"\frac{31}{6}"),
    (r"$$6-\frac{5}{6}=\frac{31}{6}$$", r"\frac{31}{6}"),
    ("Final answer: -5", "-5"),
    ("\u22125", "-5"),
    (r"\(0.5\)", r"\frac{1}{2}"),
    ("125 mod 7 = 6", "6"),
    ("Final answer: C", r"\text{C}"),
    ("0.5", "Therefore, the answer is $\\frac{1}{2}$."),
    ("<worker_scratchpad>Wrong guess: 9</worker_scratchpad><worker_result>0.5</worker_result>", r"\frac{1}{2}"),
    ("0.32", r"32 \%"),
    (r"\frac{8}{25}", r"32 \%"),
    ("32%", "0.32"),
    (r"32 \%", r"\frac{8}{25}"),
    (r"\boxed{32\%}", "0.32"),
    (r"\(32\%\)", "0.32"),
    (r"32\text{\%}", "0.32"),
    ("0.32", r"32\mathrm{\%}"),
    ("0", r"0\%"),
    ("1", r"100\%"),
    ("0.005", r".5\%"),
    ("-0.325", r"-32.5\%"),
    ("0.32", r"+32\%"),
])
def test_equivalent_answers(prediction, reference):
    assert rewarding.compute_final_answer_correctness(prediction, reference) == 1.0


@pytest.mark.parametrize("prediction,reference", [
    ("-5", "5"),
    ("5", "-5"),
    ("18", "4"),
    ("15015 % 16 = 15", "7"),
    ("8", "8+4i"),
    ("[6, \u221e)", r"(6,\infty)"),
    ("(3/5, 2/5)", r"(2/5, 3/5)"),
    ("686", r"686\pi"),
    ("3\nFinal answer: 4", "3"),
    ("Answer: 3\nFinal answer: 4", "3"),
    (r"\boxed{3}" + "\nFinal answer: 4", "3"),
    (r"\boxed{3}, but the final answer is 4.", "3"),
    (r"\boxed{3}\nFinal answer: 4", "3"),
    (r"\boxed{3} Some other answer follows.", "3"),
    (r"\boxed{3} \boxed{4}", "3"),
    ("3\n4", "4"),
    ("3 or 4", "4"),
    ("3", "x=2, y=3"),
    ("3", "x=-3 or x=3"),
    ("-3", "-3, 3"),
    ("3", "{2, 3}"),
    ("8 + 4i = 8", "8+4i"),
    ("<worker_scratchpad>4</worker_scratchpad>", "4"),
    (r"\boxed{\frac{1}{2}", r"\frac{1}{2}"),
    ("", "0"),
    ("32", r"32 \%"),
    ("0.33", r"32 \%"),
    ("32%", "32"),
    (r"32\text{\%}", "32"),
    ("32.5%", r"32\%"),
    ("3% or 32%", r"32\%"),
])
def test_wrong_or_ambiguous_answers_do_not_match(prediction, reference):
    assert rewarding.compute_final_answer_correctness(prediction, reference) == 0.0


@pytest.mark.parametrize("answer", [
    r"32 \%", "32%", r"$32 \%$", r"\(32 \%\)",
    r"32\text{\%}", r"32\mathrm{\%}",
])
def test_numeric_percentages_are_normalized_before_parsing(answer):
    parent = rewarding.__package__.rpartition(".")[0]
    backend = importlib.import_module(f"{parent + '.' if parent else ''}utils.reward_score.math_verify")
    assert backend._normalize_notation(answer) == r"\frac{32}{100}"


def test_percentage_normalization_preserves_modulo_expression():
    parent = rewarding.__package__.rpartition(".")[0]
    backend = importlib.import_module(f"{parent + '.' if parent else ''}utils.reward_score.math_verify")
    assert backend._normalize_notation("15015 % 16") == "15015 % 16"


def test_unparseable_reference_is_not_silently_scored_as_wrong(monkeypatch):
    parent = rewarding.__package__.rpartition(".")[0]
    backend = importlib.import_module(f"{parent + '.' if parent else ''}utils.reward_score.math_verify")
    monkeypatch.setattr(backend, "_verifier", lambda: (lambda *a, **kw: [], None, lambda: None))
    with pytest.raises(ValueError, match="Cannot parse ground-truth"):
        backend.compute_answer_score("0.32", "unparseable reference")


@pytest.mark.parametrize("package,module", [
    ("hierarchical_rema", "utils.reward_score"),
    ("verl.hierarchical_rema", "verl.utils.reward_score"),
])
def test_verifier_import_supports_hpc_and_package_layouts(monkeypatch, package, module):
    calls = []
    sentinel = object()
    monkeypatch.setattr(rewarding, "__package__", package)
    monkeypatch.setattr(rewarding, "_DEFAULT_COMPUTE_SCORE_LOADED", False)
    monkeypatch.setattr(rewarding, "_DEFAULT_COMPUTE_SCORE", None)
    def load(name):
        calls.append(name)
        return SimpleNamespace(_default_compute_score=sentinel)
    monkeypatch.setattr(rewarding.importlib, "import_module", load)
    assert rewarding._load_default_compute_score() is sentinel
    assert calls == [module]


def test_import_failure_is_not_silently_downgraded(monkeypatch):
    monkeypatch.setattr(rewarding, "_DEFAULT_COMPUTE_SCORE_LOADED", False)
    def missing(name):
        raise ImportError("missing backend")
    monkeypatch.setattr(rewarding.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="silent exact-match"):
        rewarding.compute_final_answer_correctness("0.5", r"\frac{1}{2}")


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_scores_fail_explicitly(monkeypatch, value):
    monkeypatch.setattr(rewarding, "_load_default_compute_score", lambda: lambda **kw: value)
    with pytest.raises(RuntimeError, match="nonfinite"):
        rewarding.compute_final_answer_correctness("0.5", r"\frac{1}{2}")


def test_backend_failure_is_not_treated_as_incorrect(monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("verifier dependency unavailable")
    monkeypatch.setattr(rewarding, "_load_default_compute_score", lambda: broken)
    with pytest.raises(RuntimeError, match="dependency unavailable"):
        rewarding.compute_final_answer_correctness("0.5", r"\frac{1}{2}")


def test_full_solution_api_remains_available():
    parent = rewarding.__package__.rpartition(".")[0]
    backend = importlib.import_module(f"{parent + '.' if parent else ''}utils.reward_score")
    assert backend._default_compute_score(
        "ReMA-math", r"We calculate it. The answer is $\frac{1}{2}$.", r"\frac{1}{2}"
    ) == 1.0
