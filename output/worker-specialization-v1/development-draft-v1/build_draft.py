#!/usr/bin/env python3
"""Author a pending development review; never certify or freeze it."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

AUTHOR = "OpenAI assistant / development-draft-v1 / 2026-10-30"
SELECTION = "c5baddf888a22c799519b83aae2cbafd2cbf7f7d86ccb1e172af7cbb19fa853e"
PLANS = {}
FLAGS = {}
NOTES = {}


def task(instruction, answer, check, kind="compute", alternatives=()):
    return dict(instruction=instruction, kind=kind,
                reference_answers=[answer, *alternatives],
                checked_result=r"LOCAL_RESULT: \boxed{" + answer + "}",
                reference_check=check)


def plan(number, *tasks, flags=(), notes=""):
    PLANS[number] = list(tasks)
    FLAGS[number] = list(flags)
    NOTES[number] = notes


plan(1,
    task("Let b be the positive integer number of coins in each of seven original bags. After finding 53 more coins, all coins can be divided equally among eight bags. Determine the residue class of b modulo 8 from this divisibility condition. Return the least nonnegative residue.",
         "5", "7b+53 is divisible by 8; 7=-1 and 53=5 modulo 8, hence b=5 modulo 8."),
    task("S1 gives the least nonnegative residue c of b modulo 8. The positive integer b satisfies b=c modulo 8, and the total 7b+53 is strictly greater than 200. Find the smallest possible b.",
         "29", "The inequality gives b>21. The first integer greater than 21 congruent to 5 modulo 8 is 29."),
    task("S2 is the smallest feasible number b of coins in each original bag. There were seven original bags. Calculate their total number of coins before the extra bag was found.",
         "203", "The original total is 7*29=203; the enlarged total is 256, divisible by 8."))

plan(2,
    task("A cone has base area B=30 square units and height h=6.5 units. Calculate the product B*h, in cubic units.",
         "195", "30*(13/2)=195."),
    task("S1 is the product P=B*h in cubic units for a cone. Its volume is V=P/3. Calculate the volume in cubic units.",
         "65", "195/3=65."))

plan(3,
    task("For symbols a and b, expand and simplify a(2a+b)-2a^2+ab into a single monomial. Return that monomial.",
         "2ab", "Expansion gives 2a^2+ab-2a^2+ab=2ab."),
    task("S1 is an integer expression in a and b. Here a and b are distinct prime numbers strictly greater than 2. Determine the number of positive integer divisors of the expression in S1.",
         "8", "2,a,b are three distinct primes, each to the first power. The divisor count is (1+1)^3=8."))

plan(4,
    task("For y=(4x^3+2x-4)/(3x^3-2x^2+5x-1), determine the y-value of the horizontal asymptote as x tends to positive or negative infinity. Return an exact number.",
         r"\frac{4}{3}", "Both polynomials have degree three; dividing by x^3 gives the limit 4/3."),
    flags=("terminal_only",),
    notes="Use one natural task rather than manufacture intermediate steps. This question contributes no nonterminal reasoning observations.")

plan(5,
    task("In the Cartesian plane let A=(2,-6) and B=(-4,3). Compute the displacement B-A. Return the ordered pair (change in x, change in y), in units.",
         "(-6,9)", "(-4-2,3-(-6))=(-6,9)."),
    task("S1 is a Cartesian displacement (u,v), with both components in units. Compute u^2+v^2, the squared Euclidean distance in square units.",
         "117", "(-6)^2+9^2=36+81=117."),
    task("S2 is a squared Euclidean distance d^2 in square units. Return the nonnegative distance d in units, expressed in simplest radical form.",
         r"3\sqrt{13}", "sqrt(117)=sqrt(9*13)=3sqrt(13)."))

plan(6,
    task("Two circles of radius 1 have centers (-1/4,0) and (1/4,0). Find the radius R of the largest circle lying inside and internally tangent to both circles. Justify where its center lies and return R as an exact fraction.",
         r"\frac{3}{4}", "Equal tangency distances force the center onto x=0. Its distance to either center is sqrt(1/16+y^2), minimized at y=0. R=1-1/4=3/4."),
    task("Two unit circles have centers (-1/4,0) and (1/4,0). A circle centered at (0,0) has radius R supplied by S1. Another circle of radius r>0 lies inside and is internally tangent to both unit circles and externally tangent to the circle of radius R. By symmetry take its center (0,y) with y>0. Use the tangency conditions to determine r as an exact fraction.",
         r"\frac{3}{28}", "External tangency gives y=R+r. Internal tangency gives 1/16+(R+r)^2=(1-r)^2. With R=3/4, r=(1-1/16-R^2)/(2(1+R))=3/28."))

plan(7,
    task("Factor 8! into primes. Return the ordered tuple of exponents (e2,e3,e5,e7) such that 8!=2^e2*3^e3*5^e5*7^e7.",
         "(7,2,1,1)", "Legendre counts give e2=4+2+1=7, e3=2, e5=1, e7=1."),
    task("S1 gives exponents (e2,e3,e5,e7) for a target digit product. Find a multiset of decimal digits from 2 through 9 whose product is 2^e2*3^e3*5^e5*7^e7. First minimize the number of digits, then choose the lexicographically smallest ascending digit tuple among all such multisets. Explain minimality and return that tuple. Digits 0 and 1 cannot improve the least positive integer with this nonzero product.",
         "(2,5,7,8,8,9)", "Digits 5 and 7 are forced. The remaining product is 1152, greater than 9^3, so at least four more digits are needed. Exhaustive sorted-multiset enumeration at six digits gives (2,5,7,8,8,9) as the lexicographically smallest solution."),
    task("S2 gives the ascending digit tuple for the least positive integer with the required digit product. Concatenate its digits in that order and return the resulting decimal integer.",
         "257889", "Concatenating (2,5,7,8,8,9) gives 257889; the digit product is 40320=8!.", kind="synthesize"),
    flags=("minimality_proof_requires_reading_reasoning",))

plan(8,
    task("The Cartesian circle x^2-10x+y^2-6y=30 can be written (x-h)^2+(y-k)^2=r^2. Complete the squares and return r^2.",
         "64", "Adding 25+9 to both sides gives (x-5)^2+(y-3)^2=64."),
    task("S1 gives the squared radius r^2 of a circle. Calculate its area A=pi*r^2. Return the exact area as a multiple of pi.",
         r"64\pi", "The area is pi*64=64pi."),
    flags=("semantic_overlap_with_prior_diagnostic_example",),
    notes="This circle appeared in earlier diagnostic discussion. Exact-string pilot exclusion does not establish semantic novelty. Retain transparently as development, not untouched evidence.")

plan(9,
    task("A jar contains 14 green marbles. A uniformly random draw is green with probability 2/15. Find the total number N of marbles in the jar.",
         "105", "14/N=2/15 gives N=14*15/2=105."),
    task("A jar has 21 blue, 40 red and 14 green marbles, with these colors mutually exclusive. Compute the total number of nonyellow marbles.",
         "75", "21+40+14=75."),
    task("S1 is the total number of marbles in a jar; S2 is the total number of its nonyellow marbles. Every marble is either yellow or nonyellow. Compute the number of yellow marbles.",
         "30", "105-75=30."))

plan(10,
    task("Five equal-price packs of bacon cost 10 dollars altogether. A pack of chicken costs twice as much as one pack of bacon. Calculate the cost in dollars of six packs of chicken.",
         "24", "The bacon unit price is 10/5=2; chicken costs 4 per pack, so six packs cost 24."),
    task("Three packs of strawberries cost 4 dollars each. Seven packs of apples each cost half as much as one strawberry pack. Calculate the combined fruit cost in dollars.",
         "26", "3*4+7*(4/2)=12+14=26."),
    task("Kelly has a budget of 65 dollars. Her bacon costs 10 dollars, S1 is her chicken cost in dollars, and S2 is her fruit cost in dollars. These are all her purchases. Calculate the money remaining.",
         "5", "65-10-24-26=5."))

plan(11,
    task("A restaurant has 8 high chairs and five times as many regular chairs as high chairs. Find its current number of regular chairs.",
         "40", "5*8=40."),
    task("A function needs one regular chair for each of 42 adults and one high chair for each of 15 babies. S1 is the current regular-chair count, and 8 high chairs are available. The chair types are not interchangeable. Compute the total additional chairs needed across both types.",
         "9", "The deficits are 42-40=2 regular chairs and 15-8=7 high chairs; total 9."))

plan(12,
    task("A school has 60 girls and twice as many boys as girls. Compute the total number of students.",
         "180", "60+2*60=180."),
    task("S1 is the total student count. There are exactly five students per teacher. Calculate the number of teachers.",
         "36", "180/5=36."))

plan(13,
    task("For this problem, interpret a third-floor balcony as three complete story heights above the ground. Each story is 24 feet high. Calculate the initial drop height in feet.",
         "72", "Using the benchmark's three-story-height convention, 3*24=72 feet."),
    task("S1 is a ball's initial drop height H in feet. Each bounce reaches 2/3 of the height from which the ball fell immediately before that bounce. Calculate the peak height after the second bounce, in feet.",
         "32", "H*(2/3)^2=72*4/9=32 feet."),
    flags=("floor_height_convention_requires_approval",),
    notes="The source reference treats 'third floor' as 3*24 feet. Other floor numbering conventions differ. This assumption is explicit in the task and must be approved or the question excluded with reason.")

plan(14,
    task("A bamboo plant is currently 20 feet tall and the target height is 600 inches. Using 12 inches per foot, compute the additional height needed, in inches.",
         "360", "600-20*12=360 inches."),
    task("S1 is the remaining height to grow in inches. Assume the plant grows at a constant rate of 30 inches per day, the rate used in this problem's numerical interpretation. Calculate the number of days needed.",
         "12", "360/30=12 days at the stated constant rate."),
    flags=("up_to_rate_ambiguity_requires_approval",),
    notes="The original question says 'up to 30 inches a day', which alone supplies only a lower bound on the required time. The benchmark answer assumes that rate is attained throughout. A reviewer must approve the convention or exclude, not silently certify uniqueness.")

plan(15,
    task("A 100-centimeter ribbon is cut into four equal pieces. Calculate the length of one piece in centimeters.",
         "25", "100/4=25 centimeters."),
    task("S1 is the length in centimeters of one ribbon piece. Each such piece is cut into five equal pieces. Calculate the final piece length in centimeters.",
         "5", "25/5=5 centimeters."))

plan(16,
    task("Melanie saves 10 toothpicks per week for 12 weeks. Calculate the number already saved.",
         "120", "10*12=120."),
    task("S1 is the number of toothpicks already saved. The target is 200, and Melanie continues saving 10 per week. Find the additional number of whole weeks required to reach the target.",
         "8", "(200-120)/10=8 exactly."))

plan(17,
    task("Timothy's locker has volume 24 cubic inches. Zack's locker has half that volume. Calculate Zack's locker volume in cubic inches.",
         "12", "24/2=12 cubic inches."),
    task("S1 is Zack's locker volume in cubic inches. Peter's locker has one quarter of Zack's volume. Calculate Peter's locker volume in cubic inches.",
         "3", "12/4=3 cubic inches."))

plan(18,
    task("A payphone charges 0.25 dollars per minute for the first 16 minutes of a call. Calculate the total charge for those first 16 minutes, in dollars.",
         "4", "16*(1/4)=4 dollars."),
    task("A payphone call lasts 36 minutes. Minutes after the first 16 cost 0.20 dollars each. Calculate the charge in dollars for the portion after the first 16 minutes.",
         "4", "(36-16)*(1/5)=4 dollars."),
    task("S1 is the charge for the first portion of a phone call and S2 is the charge for the remaining portion, both in dollars. These portions cover the whole call without overlap. Calculate the total charge.",
         "8", "4+4=8 dollars."),
    flags=("equal_intermediate_values_are_legitimate",),
    notes="Two different calculations both return 4. Their equality is not evidence of copying or a role violation.")

plan(19,
    task("A train starts with 172 passengers. At its first stop 47 get off and 13 get on. Calculate the passenger count immediately after that stop.",
         "138", "172-47+13=138."),
    task("S1 is the train's passenger count after its first stop. At its second stop 38 passengers get off and nobody gets on. Calculate the remaining passenger count.",
         "100", "138-38=100."))

plan(20,
    task("Physical therapy lasts six weeks, with two sessions per week and two hours per session. Compute the total number of therapy hours.",
         "24", "6*2*2=24 hours."),
    task("S1 is the total number of physical-therapy hours. Each hour costs 125 dollars. Calculate the total cost in dollars.",
         "3000", "24*125=3000 dollars."))

plan(21,
    task("A soccer event has 12 players and 3 coaches. Each player brings two additional guests, and nobody else attends. Calculate the total number of people to feed.",
         "39", "12+3+12*2=39."),
    task("S1 is the number of people to feed. One pizza serves three people. Calculate the minimum whole number of pizzas needed.",
         "13", "ceil(39/3)=13."),
    task("S2 is the number of pizzas to buy. Each pizza costs 15 dollars. Calculate the total cost in dollars.",
         "195", "13*15=195 dollars."))

plan(22,
    task("Convert the three temperatures 1100, 695 and 1200 degrees Celsius to kelvin using T_K=T_C+273.15. Return the ordered tuple (T1100,T695,T1200) with exact decimal entries.",
         "(1373.15,968.15,1473.15)", "Adding 273.15 gives (1373.15,968.15,1473.15) kelvin."),
    task("S1 gives the ordered temperatures (T1,T2,T3) in kelvin. Compute u=(1/T1-1/T3)/(1/T2-1/T1). Use the supplied decimals exactly and return u as an exact fraction, without rounding.",
         r"\frac{387260}{2386503}", "Exact rational arithmetic gives u=387260/2386503, approximately 0.162271. All denominators are nonzero."),
    task("Assume diffusivity follows the Arrhenius model D(T)=A*exp(-B/T), with T in kelvin and constants A,B. The measured diffusivities are D(T1)=10^-5 and D(T2)=10^-6 in cm^2/s. S2 supplies u=(1/T1-1/T3)/(1/T2-1/T1). Eliminate A and B to compute D(T3). Return n times 10^x in cm^2/s, with mantissa n rounded to two decimal places.",
         r"1.45\times 10^{-5}", "ln(D3/D1)=u*ln(D1/D2), hence D3=10^-5*10^u=1.453017698741008e-5 cm^2/s. Rounded mantissa: 1.45.", alternatives=("0.0000145",)),
    flags=("Arrhenius_model_assumption_requires_approval", "scientific_notation_preflight_required"),
    notes="Two diffusivity measurements do not define arbitrary D(T). The conventional Arrhenius assumption used by the reference solution is explicitly provided; a reviewer must approve this domain assumption. 273.15 K conversion and the source's 273 K approximation agree at the requested precision.")

plan(23,
    task("The provided equilibrium calculation states: industry supply equals demand when 100*sqrt(p)=6400/sqrt(p), giving p=64. Extract and return the stated equilibrium price p, without recomputing it.",
         "64", "The original question itself explicitly supplies equilibrium price 64. This is an intentional extraction control, not new mathematical computation.", kind="forward"),
    task("Each competitive firm's nonnegative output at price p is y=sqrt(p). S1 supplies the equilibrium price. Calculate the output of one firm.",
         "8", "The nonnegative square root of 64 is 8."),
    flags=("original_question_contains_solved_subproblems", "forward_control_not_primary_compute"),
    notes="Q already contains worked supply and equilibrium-price solutions. Forwarding the supplied price is correct assigned behavior, not a violation. The primary compute/nonterminal endpoint excludes this forward task; retain it as a separate control.")

plan(24, flags=("proposed_exclusion_missing_material_constants",),
    notes="Propose exclusion, subject to independent review: the original statement does not specify the wavelength, thorium lattice parameter or a complete set of material constants from which to obtain them. The reference uses external physical inputs. Do not invent them, import computed answers as given data, or exclude merely because the problem is difficult. No tasks drafted for this row.")

plan(25,
    task("A full town has d distinct dishes, with integer d>=2. Each resident knows precisely two distinct dishes, and for every unordered pair of dishes there is exactly one resident who knows that pair. Derive the number of residents as an expression in d.",
         r"\frac{d(d-1)}{2}", "Residents are in bijection with unordered pairs of distinct dishes: choose(d,2)=d(d-1)/2."),
    task("S1 gives a full town's population as a function of its number d of distinct dishes. Evaluate it at d=17 and return the population as an integer.",
         "136", "17*16/2=136."))

plan(26,
    task("Let k>=1 be an integer. A dictionary contains exactly the length-k strings over {A,B} that start with A, except for the all-A string, which is omitted. Count its strings and justify why no k by k grid can have every row and column in this dictionary. Return the dictionary's size as an expression in k.",
         "2^{k-1}-1", "There are 2^(k-1) strings starting with A; omit one. Every column of a putative grid would start with A, forcing an all-A first row, which is forbidden."),
    task("Encode A as 0 and B as 1. A dictionary contains a length-k binary string s and its bitwise complement. Find a symmetric bilinear polynomial f(u,v) on bits u,v such that f(0,v)=v and f(1,v)=1-v. Explain why setting grid entry M_ij=f(s_i,s_j) makes every row and column equal to s or its complement. Return f(u,v).",
         "u+v-2uv", "The polynomial u+v-2uv is symmetric and has the required two restrictions. For a fixed row or column bit it yields either s or its complement."),
    task("For integer k>=1, a dictionary D consists of distinct length-k strings over {A,B}. Find the smallest m guaranteeing a k by k grid whose rows and columns all belong to D whenever |D|>=m. Use the obstructing dictionary size in S1 and the complementary-string construction in S2. Also account for a dictionary containing an all-A or all-B string. Justify both the lower bound and the guarantee, and return m as an expression in k.",
         "2^{k-1}", "S1 requires m>=2^(k-1). A constant string gives a constant grid. Without either constant, remaining strings form 2^(k-1)-1 complementary pairs, so 2^(k-1) strings force a pair; S2 constructs the grid."),
    flags=("proof_reasoning_audit_required",),
    notes="Correct boxed expressions do not certify the obstruction or constructive proof. Audit the generated reasoning. The numerical verification checks all dictionaries for k<=3 only, not a proof for every k.")

plan(27,
    task("Write every positive integer from 1 through 2019 in ordinary decimal notation, with no leading zeros. Erase all occurrences of digits 0 and 5. Count occurrences of each retained digit. Return the ordered tuple of counts for digits (1,2,3,4,6,7,8,9), in that order.",
         "(1612,622,602,602,602,602,602,602)", "Direct enumeration of the 6969 written digits gives counts 1:1612, 2:622, and 3,4,6,7,8,9:602 each. Erased counts are 0:521 and 5:602."),
    task("S1 supplies the ordered occurrence counts (c1,c2,c3,c4,c6,c7,c8,c9) of retained decimal digits. Form their product P=1^c1*2^c2*3^c3*4^c4*6^c6*7^c7*8^c8*9^c9. Determine the prime exponents (e2,e3,e7) in P=2^e2*3^e3*7^e7, and return that ordered tuple.",
         "(4234,2408,602)", "e2=c2+2c4+c6+3c8=4234; e3=c3+c6+2c9=2408; e7=c7=602."),
    task("S2 gives (e2,e3,e7) for a positive integer P=2^e2*3^e3*7^e7. Compute P modulo 1000 and return its least nonnegative residue.",
         "976", "Modular exponentiation of 2^4234*3^2408*7^602 gives 976 modulo 1000. Direct multiplication of retained digits modulo 1000 independently agrees."))

plan(28,
    task("Evaluate cos(15 degrees)/3 exactly using angle identities. Return a simplified radical expression.",
         r"\frac{\sqrt{6}+\sqrt{2}}{12}", "cos(45-30)=(sqrt(6)+sqrt(2))/4; dividing by 3 gives the reference."),
    task("S1 is the value v of sin(x), where x is strictly between 0 and 90 degrees. Compute x=arcsin(v) in degrees and round it to the nearest tenth of a degree.",
         "18.8", "asin(cos(15 degrees)/3)=18.782423310296114 degrees; the interval selects the acute solution."))

plan(29,
    task("Let T=55. Calculate T-5 and 4T+11. Return the ordered pair (T-5,4T+11).",
         "(50,231)", "55-5=50 and 4*55+11=231."),
    task("S1 gives an ordered pair (b,c). Find K such that 20,b,K is an increasing geometric sequence and 19,K,c is an increasing arithmetic sequence. Check both conditions and return K.",
         "125", "The geometric condition gives K=50^2/20=125; the arithmetic condition gives (19+231)/2=125. Both sequences are increasing."))

plan(30,
    task("A real sequence starts a_0=0,a_1=1. For each j>=2, a_j is the arithmetic mean of some nonempty suffix a_{j-k},...,a_{j-1}, with 1<=k<=j. Fix n>=2 and let S=sum_{i=1}^{n-1} a_i. Determine the smallest and largest possible suffix averages available for choosing a_n, for any allowed preceding sequence. Prove the bounds using the averaging rule and return the ordered pair (minimum,maximum) as expressions in S,n.",
         r"(\frac{S}{n},\frac{S}{n-1})", "The extrema are attained by averaging all preceding entries, and by omitting only a_0. One can prove their extremality inductively using nested ranges of suffix averages. This is a substantive lemma, not certified by box matching or finite enumeration alone."),
    task("Let N>=2 and a_0=0,a_1=1; every later term is the mean of a nonempty suffix of preceding terms. Set S=sum_{i=1}^{N-1}a_i. S1 gives the extrema of available suffix averages at any index n in terms of the sum preceding that index. Apply those bounds to a_N and a_{N+1}, then eliminate a_N to obtain a_{N+1}-a_N<=C(N)*S. Return the resulting coefficient C(N).",
         r"\frac{1}{N^2}", "a_N>=S/N and a_{N+1}<=(S+a_N)/N. Thus the jump is at most S/N-(N-1)a_N/N, at most S/N^2."),
    task("For integer N>=2, construct a_0=0 and a_1=...=a_{N-1}=1, then choose a_N as the mean of a_0 through a_{N-1}, and a_{N+1} as the mean of a_1 through a_N. Verify that these choices obey the rule of averaging a nonempty suffix of preceding terms. Compute a_{N+1}-a_N as an expression in N.",
         r"\frac{N-1}{N^2}", "Keeping 1 uses a one-term suffix; the last two specified means are legal suffixes. a_N=(N-1)/N and a_{N+1}=1-1/N^2, so their difference is (N-1)/N^2."),
    task("A sequence starts a_0=0,a_1=1 and each later term is the mean of a nonempty suffix of preceding terms. All terms therefore lie in [0,1]. S2 supplies C(N) in the upper bound a_{N+1}-a_N<=C(N)*S, where S=sum_{i=1}^{N-1}a_i. S3 supplies an attainable jump. Combine the bound S<=N-1 with that construction to find the maximum of a_2018-a_2017. Return an exact fraction.",
         r"\frac{2016}{2017^{2}}", "S<=N-1 and S2 bound the jump by (N-1)/N^2; S3 attains it. Set N=2017."),
    flags=("proof_reasoning_audit_required", "difficult_symbolic_reference_preflight_required"),
    notes="The general extrema lemma needs an independent mathematical review. Cross-checked with official IMO 2018 Shortlist A4, Solution 2: https://www.imo-official.org/problems/IMO2018SL.pdf . Small-case enumeration is supplementary, not a universal proof. References and this source note are not worker inputs.")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def build(source_path, selection_path, output_dir):
    source_bytes = source_path.read_bytes()
    original = json.loads(source_bytes)
    selection = json.loads(selection_path.read_text())
    assert original["split"] == "development"
    assert original["selection_sha256"] == selection["selection_sha256"] == SELECTION
    assert digest({k: v for k, v in selection.items() if k != "selection_sha256"}) == SELECTION
    # Only development identities are accessed; reserved text is not inspected.
    selected = {r["question_id"]: r for r in selection["splits"]["development"]}
    assert len(original["examples"]) == len(selected) == 30
    assert [r["question_id"] for r in original["examples"]] == list(selected)
    result = deepcopy(original)
    for number, row in enumerate(result["examples"], 1):
        for field in ("question_id", "question", "final_answer", "data_source", "stratum"):
            assert row[field] == selected[row["question_id"]][field]
        assert row["decision"] == "pending" and not row["tasks"]
        assert not row["author"] and not row["checker"]
        row.update(author=AUTHOR, author_type="model", tasks=PLANS[number],
                   author_recommendation="exclude" if number == 24 else "include",
                   author_notes=NOTES[number], review_flags=FLAGS[number])
        if number == 24:
            row["exclusion_reason"] = "Missing external material constants needed for an unambiguous numerical diffraction angle. Subject to independent review."
        assert row["decision"] == "pending" and not any(row["checks"].values())
    result["draft_metadata"] = dict(status="model_authored_not_independently_reviewed",
        author=AUTHOR, original_review_sha256=hashlib.sha256(source_bytes).hexdigest(),
        reserved_question_contents_inspected=False,
        purpose="Development task/reference drafting only; not an approved dataset or a model evaluation.",
        independent_checker_required=True, evaluated_checkpoint_outputs_used=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "development-review.draft.json").open("x") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
        f.write("\n")
    notes = ["WORKER SPECIALIZATION: DEVELOPMENT DRAFT v1", "",
        "STATUS: MODEL-AUTHORED DRAFT. NOT INDEPENDENTLY REVIEWED OR READY TO FREEZE.",
        "Original user files unchanged. Reserved question contents not inspected.",
        "29 proposed plans; one proposed exclusion, all 30 decisions still pending.",
        "All checker fields blank and all approval checks false intentionally.", "",
        "REVIEW BEFORE RUNNING", "",
        "A distinct checker must review task sufficiency, mathematical references,",
        "prefixes, answer-free assigned calculations and coherence. Set decisions",
        "and checklist truthfully, not by bulk auto-approval. Reserved cases need",
        "independent human review under the existing protocol.", "",
        "Instructions and checked_result prefixes become target inputs. Reference",
        "answers, verification notes, flags and original final references do not.",
        "Full Q is added only for one variant of each nonterminal task. Terminal",
        "tasks run once without Q. Prior results are correct fixed references,",
        "not the output of the evaluated model. This isolates local performance;",
        "it does NOT test end-to-end collaboration or error recovery.", "",
        "Forwarding supplied input is an explicit control, excluded from the",
        "primary compute/nonterminal endpoint. Do not call it a violation merely",
        "because its answer appears in its instruction. Formula derivation and",
        "proof-dependent tasks need reasoning audits in addition to box scoring.", "",
        "SPECIAL CASES", "",
        "#4: naturally one task; terminal-only, no nonterminal evidence.",
        "#8: semantically familiar pilot example; historical development only.",
        "#13: floor-height convention (three complete story heights).",
        "#14: 'up to' growth rate needs an explicit constant-rate interpretation.",
        "#18: equal local answers from separate computations are legitimate.",
        "#22: Arrhenius law is an explicit domain assumption, not stated in Q.",
        "#23: original Q already supplies worked intermediate answers; forwarding control.",
        "#24: proposed exclusion for missing material constants, not model difficulty.",
        "#7, #26, #30: audit proofs/minimality, not only the boxed result.", "",
        "The sample has 5 MATH500, 4 gaokao2023en, 12 GSM8K, 3 minerva_math and",
        "6 olympiadbench questions. No AIME or AMC question appears in this small",
        "proportional development sample. Do not infer coverage of all subsets.",
        "These historical checkpoints differ in training budgets. This study is",
        "an exploratory paired checkpoint comparison, not causal proof of training.", "",
        "Math-Verify and GPU evaluation were NOT run locally. Cluster prepare",
        "must validate symbolic/tuple/scientific notation before loading models.", "",
        "AFTER INDEPENDENT REVIEW", "",
        "Use the original selection.json and a separately approved review file",
        "with scripts/eac3_worker_study.py freeze. This draft deliberately fails",
        "the review-signoff gate. Do not bypass it to launch GPU evaluation.", "",
        "DRAFT QUESTIONS AND TASKS", ""]
    for number, row in enumerate(result["examples"], 1):
        notes.extend(["=" * 72, f"#{number} | {row['stratum']} | {row['question_id']}",
                      "Question: " + row["question"], "Original final reference: " + row["final_answer"],
                      "Author recommendation (NOT review decision): " + row["author_recommendation"],
                      "Flags: " + ", ".join(row["review_flags"]), row["author_notes"], ""])
        for i, item in enumerate(row["tasks"], 1):
            notes.extend([f"S{i} [{item['kind']}]", "INPUT INSTRUCTION: " + item["instruction"],
                          "LOCAL REFERENCES (evaluator only): " + " | ".join(item["reference_answers"]),
                          "CHECKED PREFIX RESULT (for later tasks): " + item["checked_result"],
                          "REFERENCE CHECK (evaluator only): " + item["reference_check"], ""])
    with (output_dir / "development-review-notes.txt").open("x") as f:
        text = "\n".join(notes)
        f.write("\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n")
    print(json.dumps(dict(questions=len(result["examples"]),
                         proposed_plans=sum(bool(r["tasks"]) for r in result["examples"]),
                         tasks=sum(len(r["tasks"]) for r in result["examples"]),
                         kinds=dict(Counter(t["kind"] for r in result["examples"] for t in r["tasks"]))), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.review, args.selection, args.output)
