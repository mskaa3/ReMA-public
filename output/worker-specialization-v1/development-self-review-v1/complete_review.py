#!/usr/bin/env python3
"""Record the completed author self-review, never an independent certification."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'scripts/eac3_worker_study.py').is_file())
sys.path.insert(0, str(ROOT / 'scripts'))
import eac3_fixed_prefix_eval as evaluation
from eac3_worker_study import CHECKS

SOURCE = ROOT / 'output/worker-specialization-v1/development-draft-v1/development-review.draft.json'

# These are case-specific author judgements, made before any checkpoint outputs.
JUDGEMENTS = {
    1: 'The residue, minimum b, and original total are distinct valid calculations. Each later instruction defines its prefix result; integrality and the strict inequality are explicit. Redistribution checks: 203+53=256.',
    2: 'Two calculations, B*h and division by 3, preserve cubic units. Neither instruction supplies its assigned result. The volume 65 agrees with the original reference.',
    3: 'Algebra reduces to 2ab. Distinct primes greater than 2 make all three factors distinct; the divisor count is 8. The second task restates the prime assumptions needed without Q.',
    4: 'One natural limit calculation gives 4/3. Retain as terminal-only; do not treat it as evidence of distributed or nonterminal reasoning.',
    5: 'Displacement (-6,9), squared norm 117, and nonnegative distance 3sqrt(13) are consistent. Coordinate order and the meaning/units of each routed result are explicit.',
    6: 'Equal internal-tangency distances put centers on the perpendicular bisector. The largest circle is centered at the origin with R=3/4. Solving 1/16+(R+r)^2=(1-r)^2 gives r=3/28, a feasible positive radius. Instructions include the geometry required without Q.',
    7: 'Prime exponents and the minimal digit multiset were checked by exact factorization and exhaustive ordered-multiset enumeration. Concatenation is synthesis, not new arithmetic. S2 encodes the eventual integer as a tuple legitimately because that is its assigned task; do not label this overreach solely from recoverability.',
    8: 'Completing squares gives r^2=64, then area 64pi. This is semantically familiar from earlier diagnostics; retained for development with the flag, never described as unseen confirmatory data.',
    9: 'Probability gives 105 total, known colors give 75 nonyellow, and subtraction gives 30 yellow. Interpret the original typo "if 2/15" as "is 2/15". Instructions restate the intended probability.',
    10: 'Bacon unit price gives chicken total 24; independent fruit cost is 26; remaining budget is 65-10-24-26=5. Dollar units and all needed prices are in each instruction.',
    11: 'Existing regular chairs are 40; shortages are 2 regular and 7 high chairs. The explicit one-chair-per-person and noninterchangeable-type convention matches the reference. A worker is not required to infer these conventions from hidden earlier text.',
    12: 'Total students are 180 and teachers are 36. The second task specifies the meaning of S1 and the five-students-per-teacher ratio.',
    13: 'Include under the benchmark convention of three complete story heights, stated in S1. Drop height is 72 feet, second bounce is 72*(2/3)^2=32 feet. This is conditional on that floor-numbering convention, not a claim that the original wording uniquely fixes it.',
    14: 'Include under the benchmark interpretation of a constant attained growth rate of 30 inches/day, explicitly stated in S2. Extra growth is 360 inches, time is 12 days. The original "up to" alone implies only a minimum-time bound; retain the ambiguity flag and do not claim an unconditional exact time.',
    15: 'Successive equal cuts give 25 cm then 5 cm. Both stages are calculations with fully specified units and counts.',
    16: 'Saving 10 per week for 12 weeks gives 120; 80 still needed at 10/week takes 8 weeks. Whole-week indexing and target are explicit.',
    17: 'The two volume ratios yield 12 then 3 cubic inches. The later worker receives the correct named intermediate and its own required ratio.',
    18: 'The two disjoint phone-call portions each cost 4 dollars; their sum is 8. Equal partial values are valid independent calculations, not automatic duplication violations.',
    19: 'Passenger conservation gives 138 after the first stop and 100 after the second. S2 states that no passenger boards at the second stop, matching the intended question.',
    20: 'Six weeks times two sessions times two hours gives 24 hours; multiplying by 125 dollars/hour gives 3000 dollars.',
    21: '12 players, 3 coaches and 24 guests give 39 people, then 13 pizzas and 195 dollars. Whole-pizza rounding is explicit, though the division is exact here.',
    22: 'Include as a domain-conditioned task: explicitly use the conventional Arrhenius model, as in the source solution. No new instance-specific constants are imported. Exact kelvin conversion and reciprocal-temperature algebra yield u=387260/2386503 and D=1.453017698741008e-5, rounded to 1.45e-5. Two measurements alone would not determine an arbitrary function D(T).',
    23: 'The original Q already states p=64 and y=sqrt(p). S1 deliberately extracts supplied information and is labelled forward; it is outside the primary compute/nonterminal endpoint. S2 computes 8. The no-assigned-answer check applies to calculations, not to this legitimate extraction control.',
    24: 'Exclude for under-specified instance-specific inputs: numerical diffraction needs material constants/lattice data and wavelength information absent from Q. The source solution imports those values. Do not manufacture a self-contained task by silently turning computed/source-only values into givens. Exclusion was decided without model outputs.',
    25: 'The resident-pair bijection gives d(d-1)/2 for d>=2; evaluation at 17 gives 136. The irrelevant narrative is removed but all needed conditions remain.',
    26: 'Reviewed the obstruction and complementary-pair construction. Corrected "bilinear" to "degree at most one in each variable": f(0,v)=v would contradict strict bilinearity. Also restated the obstruction and construction in S3, since only earlier boxes, not earlier instructions, are routed. Proof reasoning still needs auditing in model outputs.',
    27: 'Direct digit enumeration and modular multiplication agree with exponent-based computation: counts (1612,622,602,602,602,602,602,602), exponents (4234,2408,602), residue 976. Inclusive bounds, digit order, no leading zeros and residue convention are explicit.',
    28: 'cos(15 degrees)/3=(sqrt(6)+sqrt(2))/12. The acute-domain inverse sine gives 18.782423... degrees, rounded to 18.8. Degrees and precision are stated in both relevant tasks.',
    29: 'The pair (50,231) yields K=50^2/20=125 and K=(19+231)/2=125; both sequences increase. S2 specifies both original constraints without needing Q.',
    30: 'Completed the general suffix-mean lemma by induction using monotone prefix means, not just finite checks. It gives a_N>=S/N and a_(N+1)<=(S+a_N)/N, hence jump<=S/N^2<=(N-1)/N^2. The prescribed suffix choices attain it. N=2017 gives the original reference. Box scoring alone cannot certify generated proofs.',
}


def main():
    original = evaluation.read_json(SOURCE)
    review = deepcopy(original)
    review['review_mode'] = 'model_self_review'
    review['draft_metadata']['status'] = 'superseded_by_explicit_development_model_self_review'
    review['draft_metadata']['independent_checker_required'] = False
    review['review_metadata'] = dict(
        status='completed_development_model_self_review', independently_reviewed=False,
        user_authorization='Allow self-review for development only',
        source_draft_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        review_date='2026-10-30', reserved_split_modified=False,
        checkpoint_outputs_used=False, human_review_claimed=False,
        use='Development pipeline testing only; not independent or confirmatory evidence.',
        assumption_policy='Reference benchmark conventions may be stated explicitly when they add no instance-specific measured constants. Conditional cases remain flagged. No exclusions based on checkpoint performance.',
        not_run=['Math-Verify in the training container', 'GPU inference', 'independent review'])
    assert len(review['examples']) == len(JUDGEMENTS) == 30
    for number, row in enumerate(review['examples'], 1):
        row['checker'] = row['author']
        row['checker_type'] = 'model'
        row['decision'] = 'exclude' if number == 24 else 'include'
        row['review_notes'] = JUDGEMENTS[number]
        row['checks'] = {key: number != 24 for key in CHECKS}
        if number == 24:
            row['exclusion_reason'] = JUDGEMENTS[number]
        else:
            row['exclusion_reason'] = ''
        row['author_notes'] = (row['author_notes'] + '\nSelf-review resolution: ' + JUDGEMENTS[number]).strip()
        if number in (13,14,22):
            row['review_flags'] = [flag.replace('_requires_approval', '_explicit_benchmark_assumption')
                                   for flag in row['review_flags']]
        if number == 24:
            row['review_flags'] = ['excluded_missing_material_constants']
        if number == 7:
            row['review_flags'].append('assigned_nonterminal_result_encodes_final_representation')
    tasks = review['examples'][25]['tasks']
    tasks[1]['instruction'] = ('Encode A as 0 and B as 1, with integer k>=1. A dictionary contains a '
        'length-k binary string s and its bitwise complement. Find a symmetric polynomial f(u,v) '
        'of degree at most one in each variable such that f(0,v)=v and f(1,v)=1-v for bits u,v. '
        'Explain why setting grid entry M_ij=f(s_i,s_j) makes every row and column equal to s '
        'or its complement. Return f(u,v).')
    tasks[2]['instruction'] = ('For integer k>=1, a dictionary D consists of distinct length-k '
        'strings over {A,B}. Find the smallest m guaranteeing a k by k grid with every row and '
        'column in D whenever |D|>=m. For the lower bound consider the dictionary of strings '
        'starting with A except the all-A string; S1 supplies its size. For the upper bound, '
        'if D contains a binary string s and its complement, S2 gives a polynomial f such that '
        'M_ij=f(s_i,s_j) constructs a valid grid (A=0, B=1). Also account for D containing an '
        'all-A or all-B string. Justify both bounds and return m as an expression in k.')
    review['examples'][29]['tasks'][0]['reference_check'] = (
        'Put L_n=S_n/n and U_n=S_n/(n-1), S_n=sum_{i=1}^{n-1}a_i. '
        'At n=2 the suffix extrema are L_2 and U_2. Inductively a_n lies in [L_n,U_n], '
        'so L_(n+1) is a convex combination of L_n and a_n and is >=L_n, while '
        'U_(n+1) is a convex combination of U_n and a_n and is <=U_n. '
        'For a suffix starting at j>=2, its mean is (S_(n+1)-S_j)/(n+1-j). '
        'Using S_j=j*L_j<=j*L_(n+1) gives the lower bound L_(n+1); '
        'using S_j=(j-1)*U_j>=(j-1)*U_(n+1) gives the upper bound U_(n+1). '
        'Suffixes starting at 0 and 1 attain L_(n+1) and U_(n+1). '
        'Thus the extrema are (S/n,S/(n-1)) for every n>=2.')
    for before, after in zip(original['examples'], review['examples']):
        for key in ('question_id','question','final_answer','data_source','stratum'):
            assert before[key] == after[key]
    directory = Path(__file__).resolve().parent
    evaluation.write_json(directory / 'development-review.json', review)
    notes = ['DEVELOPMENT MODEL SELF-REVIEW', '',
        'Status: completed by the same model author; NOT independently reviewed.',
        '29 included questions; 1 exclusion for missing material constants.',
        'The original draft and original user files are unchanged.',
        'Reserved-set human review rules remain unchanged.', '',
        'Cases 13, 14 and 22 are conditional on explicitly documented benchmark assumptions.',
        'Case 23 is a forwarding control; case 4 is terminal-only.',
        'Case 26 has corrected polynomial terminology and a self-contained terminal instruction.',
        'Case 30 now has the induction proof in evaluator-only reference_check.',
        'No checkpoint outputs were used for selection, review or exclusion.', '',
        'Math-Verify notation compatibility and GPU inference are still untested locally.',
        'The cluster prepare command will check notation before loading models.', '',
        'Each included checklist item reflects this author self-review, not a human signoff.',
        'The excluded question retains false/inapplicable checklist fields and an explicit reason.', '',
        'Review details:']
    for number, row in enumerate(review['examples'],1):
        notes.extend(['', f"{number}. {row['decision']} | {row['question_id']}", row['review_notes']])
    with (directory / 'review-notes.txt').open('x') as stream:
        stream.write('\n'.join(notes)+'\n')
    print(json.dumps(dict(decisions=dict(Counter(r['decision'] for r in review['examples'])),
                          review_mode=review['review_mode']), indent=2))


if __name__ == '__main__':
    main()
