#!/usr/bin/env python3
"""Check a pending draft mechanically and mathematically without certifying it."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import io
from itertools import combinations, combinations_with_replacement, product
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "scripts/eac3_worker_study.py").is_file())
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/verl/tests/rema_separated_trainer"))
import eac3_fixed_prefix_eval as evaluation
from eac3_worker_study import CHECKS, IDENTITY_FIELDS, validate_review
from eac3_worker_study_stats import validate_pairs
from local_results import extract_complete_boxed_answer, extract_local_result
from test_fixed_prefix_eval import production_rollout_class
import experiment_io


def fraction_text(value):
    value = F(value)
    return str(value.numerator) if value.denominator == 1 else rf"\frac{{{value.numerator}}}{{{value.denominator}}}"


def tuple_text(values):
    return "(" + ",".join(str(x) for x in values) + ")"


def validate_math(rows):
    checked = []

    def ref(q, t, result):
        assert str(result) in rows[q-1]["tasks"][t-1]["reference_answers"], (q, t, result)
        checked.append(f"q{q}:s{t}")

    residues = [b for b in range(8) if (7*b+53) % 8 == 0]
    ref(1, 1, residues[0])
    b = next(b for b in range(1, 1000) if (7*b+53) % 8 == 0 and 7*b+53 > 200)
    ref(1, 2, b)
    ref(1, 3, 7*b)
    ref(2, 1, 30*F("6.5"))
    ref(2, 2, 30*F("6.5")/3)
    for a, b in ((3,5), (5,7), (11,13)):
        n = a*(2*a+b)-2*a*a+a*b
        assert n == 2*a*b
        assert sum(n % d == 0 for d in range(1, n+1)) == 8
    ref(3, 1, "2ab")
    ref(3, 2, 8)
    ref(4, 1, fraction_text(F(4,3)))
    displacement = (-4-2, 3-(-6))
    ref(5, 1, tuple_text(displacement))
    squared = sum(d*d for d in displacement)
    ref(5, 2, squared)
    assert squared == 9*13
    ref(5, 3, r"3\sqrt{13}")
    R = 1-F(1,4)
    r = (1-F(1,16)-R*R)/(2*(1+R))
    assert F(1,16)+(R+r)**2 == (1-r)**2
    assert 0 < r < 1
    ref(6, 1, fraction_text(R))
    ref(6, 2, fraction_text(r))
    remainder = math.factorial(8)
    exponents = []
    for prime in (2,3,5,7):
        count = 0
        while remainder % prime == 0:
            remainder //= prime
            count += 1
        exponents.append(count)
    assert remainder == 1
    ref(7, 1, tuple_text(exponents))
    best = None
    for length in range(1, 7):
        candidates = [d for d in combinations_with_replacement(range(2,10), length)
                      if math.prod(d) == math.factorial(8)]
        if candidates:
            best = min(candidates)
            break
    assert best is not None
    ref(7, 2, tuple_text(best))
    ref(7, 3, int("".join(map(str, best))))
    r2 = 30+F(10,2)**2+F(6,2)**2
    ref(8, 1, r2)
    ref(8, 2, f"{r2}" + r"\pi")
    total, nonyellow = F(14)/F(2,15), 21+40+14
    ref(9, 1, total); ref(9, 2, nonyellow); ref(9, 3, total-nonyellow)
    chicken = F(10,5)*2*6
    fruit = 3*4+7*F(4,2)
    ref(10, 1, chicken); ref(10, 2, fruit); ref(10, 3, 65-10-chicken-fruit)
    regular = 5*8
    ref(11, 1, regular); ref(11, 2, max(0,42-regular)+max(0,15-8))
    ref(12, 1, 60+2*60); ref(12, 2, F(60+2*60,5))
    ref(13, 1, 3*24); ref(13, 2, 3*24*F(2,3)**2)
    ref(14, 1, 600-20*12); ref(14, 2, F(600-20*12,30))
    ref(15, 1, F(100,4)); ref(15, 2, F(100,4*5))
    ref(16, 1, 10*12); ref(16, 2, math.ceil(F(200-10*12,10)))
    ref(17, 1, F(24,2)); ref(17, 2, F(24,2*4))
    ref(18, 1, 16*F("0.25")); ref(18, 2, (36-16)*F("0.2"))
    ref(18, 3, 16*F("0.25")+(36-16)*F("0.2"))
    ref(19, 1, 172-47+13); ref(19, 2, 172-47+13-38)
    ref(20, 1, 6*2*2); ref(20, 2, 6*2*2*125)
    people = 12+3+12*2
    ref(21, 1, people); ref(21, 2, math.ceil(F(people,3)))
    ref(21, 3, math.ceil(F(people,3))*15)
    temps = tuple(F(x)+F("273.15") for x in (1100,695,1200))
    ref(22, 1, tuple_text(f"{float(x):.2f}" for x in temps))
    t1,t2,t3 = temps
    u = (1/t1-1/t3)/(1/t2-1/t1)
    ref(22, 2, fraction_text(u))
    d3 = 1e-5*10**float(u)
    ref(22, 3, f"{d3/1e-5:.2f}" + r"\times 10^{-5}")
    ref(23, 1, 64); ref(23, 2, math.isqrt(64))
    assert rows[23]["author_recommendation"] == "exclude" and not rows[23]["tasks"]
    ref(25, 1, r"\frac{d(d-1)}{2}")
    for d in range(2,20):
        assert len(list(combinations(range(d),2))) == d*(d-1)//2
    ref(25, 2, math.comb(17,2))
    ref(26, 1, "2^{k-1}-1")
    ref(26, 2, "u+v-2uv")
    for u,v in product((0,1), repeat=2):
        assert u+v-2*u*v == (u ^ v)
    ref(26, 3, "2^{k-1}")
    grid_checks = []
    for k in range(1,4):
        words = list(product((0,1), repeat=k))
        indices = {word: i for i,word in enumerate(words)}
        used_sets = set()
        for cells in product((0,1), repeat=k*k):
            lines = [tuple(cells[i*k+j] for j in range(k)) for i in range(k)]
            lines += [tuple(cells[i*k+j] for i in range(k)) for j in range(k)]
            used_sets.add(frozenset(indices[line] for line in lines))
        threshold = 2**(k-1)
        bad = {i for i,w in enumerate(words) if w[0] == 0 and any(w)}
        assert len(bad) == threshold-1 and not any(s <= bad for s in used_sets)
        for chosen in combinations(range(len(words)), threshold):
            assert any(s <= set(chosen) for s in used_sets)
        grid_checks.append(dict(k=k, dictionaries_at_threshold=math.comb(2**k,threshold)))
    counts = Counter("".join(map(str,range(1,2020))))
    ref(27, 1, tuple_text(counts[str(d)] for d in (1,2,3,4,6,7,8,9)))
    e2 = counts["2"]+2*counts["4"]+counts["6"]+3*counts["8"]
    e3 = counts["3"]+counts["6"]+2*counts["9"]
    e7 = counts["7"]
    ref(27, 2, tuple_text((e2,e3,e7)))
    modular = pow(2,e2,1000)*pow(3,e3,1000)*pow(7,e7,1000) % 1000
    direct = 1
    for n in range(1,2020):
        for d in str(n):
            if d not in "05":
                direct = direct*int(d) % 1000
    assert modular == direct
    assert sum(counts.values()) == 6969 and counts["0"] == 521 and counts["5"] == 602
    ref(27, 3, modular)
    ref(28, 1, r"\frac{\sqrt{6}+\sqrt{2}}{12}")
    v = math.cos(math.radians(15))/3
    assert math.isclose(v,(math.sqrt(6)+math.sqrt(2))/12,abs_tol=1e-14)
    ref(28, 2, f"{math.degrees(math.asin(v)):.1f}")
    pair = (55-5,4*55+11)
    ref(29, 1, tuple_text(pair))
    K = F(pair[0]**2,20)
    assert K == F(19+pair[1],2) and 20<pair[0]<K and 19<K<pair[1]
    ref(29, 2, K)
    ref(30, 1, r"(\frac{S}{n},\frac{S}{n-1})")
    ref(30, 2, r"\frac{1}{N^2}")
    ref(30, 3, r"\frac{N-1}{N^2}")
    ref(30, 4, r"\frac{2016}{2017^{2}}")
    sequences = [(F(0),F(1))]
    sequence_checks = []
    for n in range(2,8):
        extended = []
        for seq in sequences:
            available = [sum(seq[-k:])/k for k in range(1,n+1)]
            S = sum(seq[1:])
            assert min(available) == S/n
            assert max(available) == S/(n-1)
            assert all(0 <= x <= 1 for x in seq)
            extended.extend(seq+(x,) for x in available)
        sequences = extended
        if n >= 3:
            N = n-1
            max_jump = max(seq[-1]-seq[-2] for seq in sequences)
            assert max_jump == F(N-1,N*N)
        sequence_checks.append(dict(index=n, histories=len(sequences)))
    for N in (2,3,7,2017):
        aN = F(N-1,N)
        aNp1 = (N-1+aN)/N
        assert aNp1-aN == F(N-1,N*N)
    assert len(checked) == sum(len(row["tasks"]) for row in rows)
    return dict(task_reference_checks=len(checked), checked_tasks=checked,
        digit_minimum="Exhaustive nondecreasing multisets of 2..9, up to 6 digits",
        binary_grid_small_cases=grid_checks, suffix_average_small_cases=sequence_checks,
        limits="Symbolic/proof claims also use author reasoning. Finite cases and matching formulas do not independently certify a general proof.")


def verify(draft_path, source_path, selection_path, output):
    draft, original, selection = map(evaluation.read_json, (draft_path,source_path,selection_path))
    assert evaluation.digest({k:v for k,v in selection.items() if k != "selection_sha256"}) == selection["selection_sha256"]
    assert draft["selection_sha256"] == original["selection_sha256"] == selection["selection_sha256"]
    assert draft["draft_metadata"]["original_review_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert draft["split"] == "development" and len(draft["examples"]) == 30
    expected = {row["question_id"]: row for row in selection["splits"]["development"]}
    for row, old in zip(draft["examples"], original["examples"]):
        for field in (*IDENTITY_FIELDS,"stratum"):
            assert row[field] == old[field] == expected[row["question_id"]][field]
        assert row["decision"] == "pending" and row["author_type"] == "model"
        assert not row["checker"] and not row["checker_type"]
        assert all(row["checks"].get(k) is False for k in CHECKS)
        for item in row["tasks"]:
            routed = extract_local_result(item["checked_result"])
            assert routed and extract_complete_boxed_answer(routed) in item["reference_answers"]
    try:
        validate_review(selection,draft)
    except ValueError as exc:
        blocked_reason = str(exc)
    else:
        raise AssertionError("Pending draft unexpectedly passed review signoff")
    math_report = validate_math(draft["examples"])
    cases = []
    for row in draft["examples"]:
        tasks = row["tasks"]
        for i,t in enumerate(tasks):
            for mode in (["subtask_context"] if i == len(tasks)-1 else ["full_question","subtask_context"]):
                case = dict(id=f"{row['question_id']}:s{i+1}:{mode}", case_id=f"{row['question_id']}:s{i+1}",
                    question_id=row["question_id"], question=row["question"], family=row["stratum"], kind=t["kind"],
                    subtasks=[task["instruction"] for task in tasks], target=i+1,
                    previous_results=[task["checked_result"] for task in tasks[:i]],
                    reference_answers=t["reference_answers"], final_answer=row["final_answer"],
                    reference_check=t["reference_check"], question_mode=mode)
                cases.append(case)
    # In-memory schema/prompt inspection only: no dataset/config is published.
    evaluation.validate_dataset(dict(schema_version=1, provenance="UNAPPROVED development draft", examples=cases))
    validate_pairs(cases)
    rollout = production_rollout_class()
    prompt_hashes = {}
    for case in cases:
        def capture(row):
            with redirect_stdout(io.StringIO()), patch.dict(sys.modules, {
                    "verl.rema_separated_trainer.ppo.experiment_io": experiment_io}):
                return evaluation.capture_prompt(row,rollout)
        messages = capture(case)
        target = case["target"]
        user = messages[1]["content"]
        assert f"- S{target}: {case['subtasks'][target-1]}" in user
        assert "PLAN:" not in user
        assert case["reference_check"] not in user
        has_q = target < len(case["subtasks"]) and case["question_mode"] == "full_question"
        assert ("Reference problem:" in user) == has_q
        if has_q:
            assert case["question"] in user
        for i,result in enumerate(case["previous_results"],1):
            assert f"S{i} LOCAL_RESULT: {extract_local_result(result)}" in user
        for i,instruction in enumerate(case["subtasks"],1):
            if i != target:
                assert f"- S{i}: {instruction}" not in user
        changed = deepcopy(case)
        changed.update(reference_answers=["PRIVATE_LOCAL_REFERENCE"], final_answer="PRIVATE_FINAL_REFERENCE",
                       reference_check="PRIVATE_REVIEW_NOTES")
        assert capture(changed) == messages
        prompt_hashes[case["id"]] = evaluation.digest(messages)
    report = dict(status="draft_checks_passed_not_review_approval", selection_sha256=selection["selection_sha256"],
        draft_sha256=hashlib.sha256(draft_path.read_bytes()).hexdigest(),
        development_questions=30, proposed_plans=29, proposed_exclusions=1,
        task_count=sum(len(r["tasks"]) for r in draft["examples"]),
        task_kinds=dict(Counter(t["kind"] for r in draft["examples"] for t in r["tasks"])),
        prompt_count_if_approved=len(cases), responses_for_three_models_if_approved=3*len(cases),
        primary_compute_nonterminal_cases=sum(c["kind"]=="compute" and c["target"]<len(c["subtasks"])
                                             and c["question_mode"]=="full_question" for c in cases),
        review_gate_blocks_draft=True, review_gate_message=blocked_reason,
        approval_fields_unchanged=True, original_identities_preserved=True,
        reserved_question_contents_inspected=False, math_checks=math_report,
        production_prompt_checks=dict(target_instruction_present=True, upstream_results_routable=True,
            whole_plan_absent=True, question_visibility_correct=True, evaluator_metadata_does_not_change_prompt=True,
            capture_method="Actual production class definitions loaded through existing CPU AST test harness; no Ray/vLLM"),
        production_source_hashes=evaluation.source_hashes(), prompt_sha256=prompt_hashes,
        not_run=["independent mathematical review", "Math-Verify equivalence checks", "GPU inference", "model performance comparison"],
        scope="Development draft only. Does not certify answer-free instructions, sufficiency, scientific assumptions or general proofs.")
    evaluation.write_json(output,report)
    print(json.dumps({k:report[k] for k in ("status","task_count","task_kinds","prompt_count_if_approved",
                                         "responses_for_three_models_if_approved","primary_compute_nonterminal_cases")},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.draft,args.source,args.selection,args.output)
