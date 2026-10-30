#!/usr/bin/env python3
"""Reproducible model-authored mathematical review, NOT independent certification."""
import json
from pathlib import Path
import sys
from collections import Counter
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import eac3_worker_study as study
import eac3_fixed_prefix_eval as evaluation
PLANS, ISSUES, NOTES = {}, {}, {}
AUTHOR = 'OpenAI assistant / exploratory-300-v1 / 2026-10-30'

def t(instruction, answer, check, kind='compute'):
    answer = str(answer)
    answer = answer.replace(r'\frac\pi2', r'\frac{\pi}{2}')
    answer = answer.replace(r'\frac\pi4', r'\frac{\pi}{4}')
    answer = answer.replace(r'\lambda_{\text{red}}', r'\lambda_{red}')
    return dict(instruction=instruction, reference_answers=[answer], kind=kind,
                checked_result=r'LOCAL_RESULT: \boxed{' + answer + '}', reference_check=check)

def p(i, *tasks, note=''):
    assert i not in PLANS and i not in ISSUES, i
    PLANS[i] = list(tasks)
    NOTES[i] = note

def defer(i, category, reason):
    assert i not in PLANS and i not in ISSUES, i
    ISSUES[i] = dict(category=category, reason=reason)

p(1,t('Factor m^4+1/4 as a product of two monic quadratic polynomials over the rationals.',r'(m^2-m+\frac12)(m^2+m+\frac12)','Difference of squares: (m^2+1/2)^2-m^2.'),
 t('Use the factorization from S1 for f(m)=m^4+1/4. For a positive integer n, simplify the product of f(2 j)/f(2 j-1) for j=1,..., n. Return a polynomial in n.', '8n^2+4n+1','Let h(m)=m^2+m+1/2. f(m)=h(m)h(m-1); cancellation leaves h(2n)/h(0).'))
p(2,t('Expand 4(x+7)(2-x). Return the coefficient tuple (a, b, c) of ax^2+bx+c.', '(-4,-20,56)','4(-x^2-5x+14).'),t('S1 gives (a, b, c) for a quadratic ax^2+bx+c. Determine its maximum over all real x.',81,'Vertex x=-b/(2a)=-5/2; value 56+25=81.'))
p(3,t('Factor x^8+3 x^4-4 completely into non constant monic polynomials irreducible over the integers. Return their product.', '(x-1)(x+1)(x^2+1)(x^2-2x+2)(x^2+2x+2)','Set u=x^4: (u-1)(u+4); use difference of squares and Sophie Germain. Quadratic discriminants are negative.'),t('S1 lists the monic irreducible factors of a polynomial. Evaluate each factor at x=1 and add the results.',10,'0+2+2+1+5=10.'))
p(4,t('Solve 2^3*3^x=72 for real x.',2,'3^x=9.'))
p(5,t("Alice and Bob each initially have one ball of each of five colors. Alice transfers one uniformly random ball to Bob. Conditional on that color being red, how many balls are now in Bob's bag and how many are red? Return (total, red).",'(6,2)','One extra ball adds one to total and red count.'),t("S1 gives (total, red) in Bob's bag after Alice transferred red. Bob returns a uniformly selected ball. The original bags each had one ball of each of five colors. Find the probability that the bags now have identical color counts.",r'\frac13','Equality requires returning red; probability 2/6. Symmetry removes conditioning.'))
p(6,t('A cylinder has radius 3 inches and height 8 inches. Compute its volume divided by pi.',72,'r^2*h=9*8.'),t('S1 is a cylindrical container volume divided by pi. A sphere of radius 6 inches must be filled. How many full containers are needed?',4,'Sphere volume/pi=4*216/3=288; 288/72=4.'))
p(7,t('The cost of a solid cube of edge s is C(s)=A*s^2+B*s^3. Given C(1)=1.30 and C(2)=6.80 dollars, determine (A, B).','(0.9,0.4)','A+B=1.3; 4A+8B=6.8 gives B=.4,A=.9.'),t('S1 gives (A, B) for C(s)=A*s^2+B*s^3 dollars. Calculate C(3).','18.90','9*.9+27*.4=18.9.'))
p(8,t('Simplify 19 x+1-4 x-81.', '15x-80','Combine coefficients 19-4 and constants 1-81.'))
p(9,t('Compute 99^2.',9801,'(100-1)^2=10000-200+1.'),t('S1 is 99 squared. Compute S1+99+1.',9901,'9801+100=9901.'))
p(10,t('Expand 4(3 r^3+5 r-6)-6(2 r^3-r^2+4 r), and collect powers of r.','6r^2-4r-24','Cubic terms cancel; quadratic +6,linear20-24,constant-24.'))
p(11,t('HO and HN are tangents from an external point H to a circle centered at J, with O, N tangency points. Find the sum of the interior angles at O and N in quadrilateral JNHO, in degrees.',180,'Each radius is perpendicular to its tangent.'),t('S1 is the sum of two interior angles in a quadrilateral. Determine the sum of its other two interior angles, in degrees.',180,'360-180=180.'))
p(12,t('A monic quartic has roots 1, 2, 3, r. Return the ordered pair (p(0), p(4)) in terms of r.','(6r,24-6r)','p(x)=(x-1)(x-2)(x-3)(x-r).'),t('S1 gives (p(0), p(4)). Simplify their sum.',24,'6r+(24-6r)=24.'))
p(13,t('A traveler exchanges half of 300 US dollars for pounds at 1.64 USD per pound and the other half for euros at 1.32 USD per euro. Return (pounds, euros) exactly.',r'(\frac{3750}{41},\frac{1250}{11})','150/(41/25),150/(33/25).'),t('S1 gives (pounds, euros) acquired. Subtract the numerical pounds amount from the numerical euros amount and round to the nearest integer.',22,'1250/11-3750/41=10000/451=22.1729... .'))
p(14,t('Count the collinear unordered triples of distinct points in the grid {(x, y): x, y in {0, 1, 2}}.',8,'3 rows,3 columns,2 diagonals; no other slope can have three lattice points in this grid.'),t('Choose three distinct points uniformly from a 3-by-3 lattice grid. S1 counts the collinear triples. Find the probability of collinearity.',r'\frac{2}{21}','8/binom(9,3)=8/84.'))
p(15,t('In a 3-by-4 rectangle, find the area of the triangle with vertices (0, 4),(0, 0),(2, 0).',4,'Base2,height4,area4.'),t('S1 is the area of one shaded triangle. A second shaded triangle has vertices (0, 4),(3, 4),(3, 0); their interiors are disjoint. Find the total shaded area.',10,'Second area=3*4/2=6; total4+6.'))
p(16,t('Solve (3^k)^6=3^6 for real k.',1,'6k=6.'))
p(17,t('Let f(x)=2 x-3. Compute f(5)-1.',6,'10-3-1=6.'),t('Let g(x)=x+1. Evaluate g at the input supplied by S1.',7,'g(6)=7.'))
p(18,t('Positive y varies inversely with sqrt(x). Given y=15 at x=24, determine k^2 in y=k/sqrt(x).',5400,'k^2=y^2*x=225*24.'),t('S1 gives k^2 for y=k/sqrt(x) with positive k and x. Find x when y=3.',600,'x=k^2/9=600.'))
p(19,t('Solve -4<2(x-1)<8. Return the open interval of possible x.', '(-1,5)','Divide by2 and add1.'),t('S1 is an open interval (a, b). Calculate a+b.',4,'-1+5=4.'))
p(20,t('Five yaps equal three baps. Convert 42 baps to yaps.',70,'42*5/3=70.'),t('Four daps equal seven yaps. S1 is a quantity measured in yaps. Convert it to daps.',40,'70*4/7=40.'))
p(21,t('Digits T, D are distinct integers 1 through 9 and satisfy 2 T=5 D. Find (T, D).','(5,2)','D even and T multiple5; onlyT5,D2.'),t('S1 gives (T, D). Distinct digits S, C satisfy T+S=C and 2 T=C+D. Find S.',3,'C=10-2=8,S=8-5=3.'))
p(22,t('Two opposite sides of a square are shortened by 40 percent, and the other dimension is lengthened by 50 percent. Find the ratio of new rectangle area to original square area.', '0.9','.6*1.5=.9.'),t('S1 is the new-to-original area ratio. Find the percentage decrease in area.',10,'100*(1-.9)=10.'))
p(23,t('A well-mixed container holds 2 gallons of pure milk. One third is removed and replaced with water, with volume restored each time. Find the milk fraction after three repetitions.',r'\frac{8}{27}','Each round retains2/3 of milk; fraction(2/3)^3.'),t('S1 gives the fraction of milk in a 2-gallon mixture. Find the volume of milk in gallons.',r'\frac{16}{27}','2*8/27.'))
p(24,t('For grades 9, 10, 11, 12, male bus percentages are 41.1, 34.4, 20.6, 11.6 and female bus percentages are 39.4, 33.1, 13.8, 8.6. Equal numbers of males and females are in each grade. Which grade has male bus riders closest to 135 percent of female bus riders?',12,'Ratios approximately1.043,1.039,1.493,1.349; grade12 is closest.'))
p(25,t('For integer a=0, 1, 2, 3, 4, 5, count the integers b satisfying a^2+b^2<=25. Return counts in that order.','(11,9,9,9,7,1)','For fixeda, count=2floor(sqrt(25-a^2))+1.'),t('S1 counts allowable b for nonnegative a in a^2+b^2<=25. Use symmetry in a to count all ordered integer pairs (a, b).',81,'11+2*(9+9+9+7+1)=81.'))
p(26,t('In the square 0<x<3, 0<y<3, find the area where x+y<=1 or x-y>=1 or y-x>=1.',r'\frac92','Disjoint corner triangles have areas1/2,2,2.'),t('Independent x, y are uniform on (0, 3). S1 is the area of the region where lengths 1, x, y cannot form a nondegenerate triangle. Find the probability that they can.',r'\frac12','1-(9/2)/9=1/2.'))
p(27,t('Let z=1+i. Compute (|3 z+1|,|5 z+7|).','(5,13)','3z+1=4+3i,5z+7=12+5i.'),t('S1 gives the magnitudes of numerator and denominator of w=(3 z+1)/(5 z+7). Find |w|.',r'\frac5{13}','Magnitude quotient.'))
p(28,t('Solve 2/3=4/(x-5) over the reals, respecting the nonzero denominator.',11,'2(x-5)=12.'))
p(29,t('An arithmetic sequence has fourth term 200 and eighth term 500. Find the common difference.',75,'(500-200)/(8-4).'),t('The fourth term of an arithmetic sequence is 200. S1 is its common difference. Find its sixth term.',350,'200+2*75.'))
p(30,t('For the Cartesian point (0, 3), compute its positive distance r from the origin.',3,'sqrt(0^2+3^2).'),t('For the point (0, 3), find its polar angle theta in radians with 0<=theta<2*pi.',r'\frac\pi2','Positive y-axis.'),t('S1 is polar radius r and S2 is polar angle theta. Return the ordered pair (r, theta).',r'(3,\frac\pi2)','Combine the two values.',kind='synthesize'),note='Familiar diagnostic question; not semantically unseen.')
p(31,t('For P=(3,-10, 1), Q=(-5,-3, 6) and v=(2,-9,-2), compute the cross product (Q-P) cross v.','(31,-6,58)','Q-P=(-8,7,5); componentwise cross product.'),t('Two parallel lines pass through P, Q with common direction v=(2,-9,-2). S1 is (Q-P) cross v. Find their distance.',7,'|cross|^2=4361=49*89,|v|^2=89.'))
p(32,t('Simplify (5/sqrt(80)+sqrt(845)/9+sqrt(45))/sqrt(5) to a reduced rational number.',r'\frac{169}{36}','Terms1/4,13/9,3 sum169/36.'),t('S1 is a positive rational number. Take its principal square root.',r'\frac{13}{6}','sqrt(169/36).'))
p(33,t('Let z be complex with |z|=1 and t=|1+z|. Express |1-z+z^2| solely in terms of t.',r'|t^2-3|','Divide byz: 2Re(z)-1; t^2=2+2Re(z).'),t('S1 expresses the second magnitude as a function of t. Maximize t plus that expression over 0<=t<=2.',r'\frac{13}{4}','For t<=sqrt3,3+t-t^2 maximum13/4 att=1/2; other branch maximum3att2.'))
p(34,t('From the six primes 2, 3, 5, 7, 11, 13, draw twice independently and uniformly. Count ordered outcomes where both draws have the same parity.',26,'One even-even and25odd-odd.'),t('S1 counts favorable ordered outcomes out of 36. Find the probability that the sum of the two drawn primes is even.',r'\frac{13}{18}','26/36=13/18.'))
p(35,t('For f(x)=ax^4-bx^2+x+5, find f(3)-f(-3).',6,'Even-degree and constant terms cancel.'),t('S1 is f(3)-f(-3), and f(-3)=2. Find f(3).',8,'2+6.'))
p(36,t('How many real zeros does x^2-9 have?',2,'x=3,-3 are distinct.'))
p(37,t('Of 50 students, 6 take neither mathematics nor science. How many take at least one?',44,'50-6.'),t('S1 gives the number taking at least one of two subjects. 28 take mathematics and 21 science. Find the number taking both.',5,'28+21-44.'))
p(38,t('Find all residues t in {0,..., 22} such that t(t+2)=1 modulo 23. Return the set.',r'\{4,17\}','(t+1)^2=2mod23; roots of2 are5,18; subtract1.'),t('S1 contains all solutions modulo 23. Return the odd solution with 0<t<23.',17,'4is even,17odd.'))
p(39,t('Add 313.9 and 12.6.','326.5','Align decimal places.'))
defer(40,'malformed_reference','Source answer is 3\\2\\sqrt{2}; worked source gives 3 plus/minus 2sqrt(2). Inverse-cotangent branch convention also needs review. Original reference not silently changed.')
p(41,t('Divide x^9+1 by x-1. Return the polynomial quotient, not the remainder.','x^8+x^7+x^6+x^5+x^4+x^3+x^2+x+1','(x-1)*sum_{0..8}x^k=x^9-1; remainder2.'))
p(42,t('Let f(x)=3 x^4+5 x^2-9 x-2 and g(x)=f(x-1). Compute g(1).',-2,'g1=f0=-2.'),t('S1 gives g(1) for a polynomial g. Find the sum of all coefficients of g.',-2,'Coefficient sum equals valueat1.',kind='forward'))
p(43,t('Find the area in 0<=x<=8, 0<=y<=4 satisfying x+y<=4.',8,'Righttriangle legs4,4.'),t('A point is uniform in an 8-by-4 rectangle. S1 is the area of a favorable region. Find its probability.',r'\frac14','8/32.'))
p(44,t('A bake sale sells 54 cookies at 3 per dollar, 20 cupcakes at 2 dollars each and 35 brownies at 1 dollar each. Compute total revenue in dollars.',93,'54/3+20*2+35.'),t('S1 is revenue in dollars. Costs were 15 dollars. Compute profit.',78,'93-15.'))
p(45,t('Three Blinkets equal seven Drinkets. Convert 56 Drinkets to Blinkets.',24,'56*3/7.'),t('One Trinket equals four Blinkets. Convert the Blinket amount from S1 to Trinkets.',6,'24/4.'))
p(46,t('For x=3.6, return (ceil(x^2), ceil(x)^2).','(13,16)','ceil12.96=13;ceil3.6=4.'),t('Subtract the second entry of S1 from the first.',-3,'13-16=-3.'))
defer(47,'unresolved_proof','315 is supplied but not yet checked by a complete count of rectangles with vertices at diagonal intersections; counting only polygon vertices would be wrong.')
p(48,t('Positive box edges a, b, c satisfy ab+ac+bc=27 and abc=23. Show that an extremum of squared diagonal has two equal edges, and find the largest squared diagonal.',r'\frac{657}{16}','Compact constrained domain. Lagrange multipliers force two equal edges; the cubic x^3-27x+46 factors (x-2)(x^2+2x-23). Compare the two positive roots2 and2sqrt6-1, giving maximum657/16.'),t('S1 is largest squared box diagonal. An enclosing sphere has radius half a diagonal. Write the required radius squared as p/q reduced and return p+q.',721,'657/64, sum721.'))
p(49,t('Four numbers are drawn from ten. A ticket chooses four. Count draw-sets matching at least two ticket numbers.',115,'C4,2*C6,2+C4,3*C6,1+C4,4=90+24+1.'),t('Exactly one draw-set matches all four ticket numbers. Conditional on at least two matching, whose count is S1, probability is m/n reduced. Return m+n.',116,'1/115 =>116.'))
p(50,t('For coins 1, 10, 25, write N=25 q+r with integer q>=1 and 0<=r<25. Compare greedy change to optimal change. Count the residues r for which greedy is optimal, and justify the count.',15,'Removing an even number2t of quarters increases coin count by3t. Removing2t+1 changes it by3t+6 if r mod10<5, or3t-3 otherwise. Thus precisely r=5..9,15..19 fail;15 residues succeed.'),t('S1 counts successful residues per complete 25-block with at least one quarter. For N<25 greedy is always optimal. Count N in 1..1000 for which largest-first greedy change with coins 1, 10, 25 is optimal.',610,'24 amounts below25;39 complete blocks25..999 with15 successes each;1000 succeeds.24+585+1=610.'))
defer(51,'unresolved_geometry','AIME2025 circle/arc calculation needs a checked geometric derivation; the provided final value alone is insufficient to certify local references.')
defer(52,'unresolved_geometry','AIME2025 internal equidistant-point configuration requires checked coordinate branches before authoring local references.')
p(53,t('In the first quadrant, find the area defined by |x-1|+|y-1|<=1.',2,'Diamond vertices(0,1),(1,0),(2,1),(1,2),diagonals2.'),t('S1 is the area in one quadrant of ||x|-1|+||y|-1|<=1. Use reflection symmetry to find total area.',8,'Fourquadrants interiorsdisjoint;4*2.'))
p(54,t('Define (a+bi) tensor(c+di)=ac+bd*i. If z=a+bi satisfies z tensor z=z^2+40, find(a^2, b^2).','(10,40)','Real parts give b^2=40; imaginary parts b^2=2ab, hence a=b/2.'),t('S1 gives(a^2, b^2). Find |a+bi|^2.',50,'10+40.'))
p(55,t('Two travelers start 45 miles apart and move toward each other at 18 and 12 mph. Find the meeting time in hours.',r'\frac32','45/(18+12).'),t('S1 is meeting time in hours. How far has the 18 mph traveler gone?',27,'18*3/2.'))
p(56,t('A quiz average is m after n quizzes. One additional score 11 raises the average by 1; three additional scores 11 raise it by 2. Find (n, m).','(3,7)','11=m+n+1 and33=3m+2n+6 implyn3,m7.'),t('S1 gives (quiz count, current mean). Return the current mean.',7,'Read second component.',kind='forward'))
defer(57,'missing_question_context','Question says continue from previous question but omits race distance. Options suggest5miles; do not supply it as if it had been stated.')
p(58,t('Compute discriminants of 3 x^2-3, 3 x^2+3 x, 3 x^2+3 x+3, 3 x^2-6 x+3, in order.','(36,9,-27,0)','b^2-4ac.'),t('S1 gives discriminants of choices A, B, C, D. Which quadratic has no real roots?','C','Only the third discriminant is negative.',kind='synthesize'))
p(59,t('Solve u^2-4 u-12=0. Return the set of real roots.',r'\{-2,6\}','(u-6)(u+2).'),t('S1 gives allowable values of u=3 x. Find the positive x.',2,'u6/3.'))
p(60,t('Use the imaginary part of (1+it)^2023. Determine the coefficient of t^2023.',-1,'i^2023=i^3=-i.'))
p(61,t('Find the y-intercepts of f(x)=(x+2)(x-8) and g(x)=x^2+5 x-12, as an ordered pair of y-values.','(-16,-12)','Evaluateat0.'),t('Sum the two y-intercepts in S1.',-28,'-16-12.'))
p(62,t('Using modular cycles, find the units digits of 2022^2023 and 2023^2022. Return the ordered pair.','(8,9)','2cycle4 exponent3;3cycle4 exponent2.'),t('S1 gives the units digits of two integers. Find the units digit of their sum.',7,'8+9=17.'))
p(63,t('A population model starts at 4.44 billion in 1980 and grows by 0.076 billion/year. Predict 2015 population in billions.','7.10','4.44+35*.076=7.10.'),t('S1 is predicted population in billions. Actual was 7.35 billion. Find positive difference.',r'\frac14','7.35-7.10=.25.'))
p(64,t('Positive integer rectangle sides a, b obey ab-10(a+b)=2023. Determine the factorization of (a-10)(b-10) as a numerical integer and its prime factors.', '2123=11*193','Add100;2123=11*193 with193prime.'),t('Use S1 and positive integer a, b satisfying (a-10)(b-10)=2123. Find the minimum perimeter 2(a+b).',448,'Positivefactorpairs1,2123and11,193; negative factors cannot both givepositivea,b sinceeach<10 in magnitude impliesproduct<100. minsum224,perimeter448.'))
p(65,t('Which identity equals cos(A) for every positive angle A in degrees: F cos(90+A), G cos(180-A), H cos(180+A), J cos(270+A), K cos(360-A)?','K','Cos periodicity and evenness.'))
p(66,t('Solve y=7-4 x and 15 x-4 y=3 for x.',1,'15x-28+16x=3=>31x31.'),t('S1 is x. Use y=7-4 x and return (x, y).','(1,3)','y3.'))
p(67,t('A positive integer N leaves remainders 1, 2, 3 upon division by 2, 3, 4 respectively. Determine its least nonnegative residue modulo 12.',11,'N+1 divisiblebylcm2,3,4=12.'),t('S1 is the required residue modulo 12. Find the smallest N strictly greater than 48 with that residue.',59,'47notgreater48;next59.'))
p(68,t('In a survey, 53 people answered almost never and 29 answered never. Find the combined count.',82,'53+29.'))
p(69,t('A positive square n^2 is less than 2023 and divisible by 5. Find the largest admissible positive integer n.',40,'n<=44 and5|n.'),t('S1 is the largest admissible n. Count positive multiples of 5 up to that n; these each give a distinct square.',8,'40/5.'))
p(70,t('Reduce 1+...+n=(n+1)+...+(n+35) to a monic quadratic equation in n.', 'n^2-69n-1260=0','n(n+1)/2=35n+630.'),t('Solve the equation from S1 for its positive integer root.',84,'(n-84)(n+15)=0.'))
p(71,t('A pyramid with square base edge 4 has a similar top pyramid of base edge 2 and height 3. Determine the original height.',6,'Similarity ratio2.'),t('S1 gives original height. Original square base edge 4; removed top square base edge 2, height 3. Find remaining frustum volume.',28,'16*6/3-4*3/3=32-4.'))
p(72,t('Four labeled players independently choose rock, paper, scissors. Count outcomes in which one player strictly beats all three others.',12,'Choose winner4, winninggesture3; othersmustallchoosebeaten gesture.'),t('S1 counts favorable outcomes among 3^4 equally likely ones. Find probability.',r'\frac4{27}','12/81.'))
p(73,t('Points (u, log_2 u),(v, log_2 v) have midpoint(6, 2), with positive u, v. Find (u+v, uv).','(12,16)','Sum12;log2(uv)=4.'),t('S1 gives sum s and product p of two positive numbers. Find their positive difference.',r'4\sqrt5','sqrt(s^2-4p)=sqrt80.'))
defer(74,'malformed_reference','Stored answer includes trailing solution text (145solution:). Intended145 can be derived but source-label repair must be versioned explicitly.')
p(75,t('For C(x)=4.3 x+19 modeling annual soybean use in millions of tons, x years after 1995, select: A use 4.3 each year; B annual use increases 4.3 each year; C use 4.3 in 1995; D total use 4.3 over period.','B','Slope is change in annual quantity per elapsedyear.'))
p(76,t('A square is partitioned into 25 equal small squares. Its centered 3-by-3 square contains how many small squares?',9,'3*3.'),t('S1 counts small squares in a region of a 25-equal-square whole. Find percentage of area.',36,'100*9/25.'))
p(77,t('A 1200-person town has 400 people with green eyes, 600 with blue, 900 with brown; each has 2 eyes. Find total color-person incidences.',1900,'400+600+900.'),t('S1 counts distinct eye-color incidences over 1200 people. Each person contributes 1 if both eyes same color and 2 otherwise. How many have same-color eyes?',500,'Letm same; m+2(1200-m)=1900=>m500.'))
p(78,t("On each of 8 tests Gustav scored 2 points more than Russ. By how much does Gustav's average exceed Russ's?",2,'Sumdifference16/8=2.'))
p(79,t('Given f(x)=x+4, evaluate f(5).',9,'5+4.'))
p(80,t('An exponential decreases 20 percent per unit x and g(2)=16. Choose A16(.8)^(x-2), B16(.8)^(x+2), C16(.2)^(x-2), D16(.2)^(x+2).','A','Factor.8andshift2.'))
p(81,t('A model uses t for years participating in professional chess. What t describes a player entering the first professional tournament?',0,'Zero elapsedyears.',kind='forward'))
defer(82,'incorrect_reference','Righttriangle legs5,8 has area20cm^2, but stored numerical answer20^2 would be400. Do not conflate units with exponent.')
defer(83,'incomplete_or_incorrect_reference','sqrt(m^2)=sqrt(64r^2) implies |m|=8|r|; does not determine scalar64. Missing requested variable/context.')
p(84,t('P={1, 2}, Q={2, 3}. Find elements in P but not Q.',r'\{1\}','Remove common2.'))
p(85,t('For omega>0, roots of cos(omega*x)-1 on[0, 2 pi] are x=2 pi*k/omega for nonnegative integers k<=omega. Express their count using floor(omega).',r'\lfloor\omega\rfloor+1','Include k0.'),t('Using S1, determine positive omega values for which there are exactly 3 roots on[0, 2 pi].','[2,3)','flooromega2.'))
p(86,t('Choose 60 students proportionally from 400 middle and 200 high school students. Determine selected counts (middle, high).','(40,20)','Ratio2:1,total60.'),t('S1 gives counts (m, h). Express the number of ways to select those students from 400 middle and 200 high school students using binomial coefficients.',r'\binom{400}{40}\binom{200}{20}','Independentunorderedchoices.'))
p(87,t('Evaluate k(x)=260 x dollars at x=10 weeks.',2600,'260*10.'))
p(88,t('Combine 1/d+1/e+1/f into one fraction for positive d, e, f.',r'\frac{de+df+ef}{def}','Common denominatordef.'),t('S1 is 1/C for positive C. Express C in terms of d, e, f.',r'\frac{def}{de+df+ef}','Reciprocal.'))
p(89,t('Similar trapezoids have corresponding side ratio 3. Determine the area ratio.',9,'Squarelinearscale.'),t('S1 is larger-to-smaller area ratio. Smaller area 6 cm^2. Find larger area.',54,'9*6.'))
p(90,t('Simplify z=(1-i)/(2+2 i).',r'-\frac i2','Multiply numerator/denominator1-i.'),t('Using z from S1, calculate z minus its complex conjugate.', '-i','-i/2-i/2.'))
p(91,t('A dance costs 225 dollars for lights, 600 for food, 75 for decorations. Desired net profit 500. Calculate required revenue.',1400,'225+600+75+500.'),t('S1 is required revenue in dollars. 400 students attend paying equally. Find ticket price in dollars.','3.50','1400/400.'))

# Grade-school arithmetic: references recomputed here, not taken from worker outputs.
p(92,t('Peyton does yoga for 1 hour. Tuesday cooking lasts three times as long; Thursday museum tour lasts half the cooking time. Find combined cooking and museum hours.', '4.5','3+1.5.'),t('S1 gives cooking plus museum hours. Add 1 hour yoga,.5 hour cheese tasting and 2 hours errands. Find total hours.',8,'4.5+1+.5+2.'))
p(93,t('A tank depth is 17 feet Monday and 7 feet greater Tuesday. Find Tuesday depth.',24,'17+7.'),t('Wednesday depth is two thirds of Tuesday depth from S1. Find Wednesday depth in feet.',16,'24*2/3.'))
p(94,t('Dolly owns 2 books and Pandora 1 different book. How many different books are available?',3,'2+1.'),t('S1 gives number of different books available. Each of two people reads all of them once. Count total book-readings.',6,'2*3.'),note='Count reading events, not distinct titles; benchmark uses this interpretation.')
p(95,t("Sofie has 25 more books than Anne; Anne has 12 fewer than Fawn. Together 85. Find Anne's count.",16,'A+(A+25)+(A+12)=85=>3A48.'),t("S1 is Anne's book count. Fawn has 12 more. Find Fawn's count.",28,'16+12.'))
p(96,t('A museum is 150 miles from home. Round trip speed 75 mph. Calculate total driving hours.',4,'2*150/75.'),t('S1 is total driving hours. Add 6 hours at museum to find total absence.',10,'4+6.'))
p(97,t('A bridge limit is 5000 pounds; empty truck plus driver 3755 pounds. Find remaining carrying capacity.',1245,'5000-3755.'),t('S1 is capacity in pounds. Each box weighs 15 pounds. Find maximum integer box count.',83,'floor1245/15=83.'))
p(98,t('A shirt costs 25 dollars and shorts 35 dollars before a 30 percent discount on both. Compute discounted total.',42,'(25+35)*.7.'),t('S1 is purchase cost including tax. Budget 50 dollars. Find money left.',8,'50-42.'))
p(99,t('Typing measurements start 47 WPM then 52 WPM. Next measurement improves by 5 WPM. Find third measurement.',57,'52+5.'),t('First two typing measurements are 47 and 52 WPM; S1 gives third. Find mean of all three.',52,'(47+52+57)/3.'))
p(100,t('A raspberry bush has 6 clusters of 20 berries. Count berries in clusters.',120,'6*20.'),t('S1 counts clustered berries. Add 67 individual berries to find total.',187,'120+67.'))
p(101,t('A farmer makes 5 hay bales per hour while a truck removes 3 per hour. Find hourly net accumulation.',2,'5-3.'),t('S1 is hourly net bales. Starting empty, how many remain after 6 hours?',12,'2*6.'))
p(102,t('Emma made 18, 21, 15 vlogs in first three weeks. Find total so far.',54,'18+21+15.'),t('S1 is completed vlogs. Target 72. Find remaining vlogs.',18,'72-54.'))
p(103,t('Ten CDs cost 15 dollars each before 40 percent discount. Find amount paid.',90,'10*15*.6.'),t('S1 is original purchase cost. Five CDs are sold for 40 dollars total. Find net money out.',50,'90-40.'))
p(104,t('Ten cars sell for 25000 dollars each; acquisition cost was 30 percent of selling price. Find total profit before commission.',175000,'10*25000*.7.'),t('S1 is profit. Salesperson receives 10 percent of it. Find commission.',17500,'*.1.'))
p(105,t("A boy has 5 cards. His brother has 3 fewer. Find brother's count.",2,'5-3.'),t("S1 is brother's count. Boy has 5. Find combined count.",7,'2+5.'))
p(106,t('A doctor charges 300 dollars/hour for 30 minutes. 30 painkillers cost 4 dollars each. Find combined visit-time and pill cost.',270,'150+120.'),t('S1 gives visit-time and pill cost. Add 200 dollar cast and 2 hour parking at 6 dollars/hour. Find total.',482,'270+200+12.'))
p(107,t('Kelly has 5 quarters worth 25 cents each and 2 dimes worth 10 cents each. Find total cents.',145,'125+20.'),t('S1 is available cents. Drink costs 55 cents. Find change.',90,'145-55.'))
p(108,t('An ice cube cools coffee by 13 degrees. How many cubes produce 65 degrees cooling?',5,'65/13.'),t('S1 gives cube count. Each adds 12 ml and 15 ml of cream is also added. Find total added liquid in ml.',75,'5*12+15.'))
p(109,t('From a 2400 dollar paycheck, 50 percent goes to retirement and 20 percent to car payment. Find total allocated.',1680,'2400*.7.'),t('S1 is allocated money out of 2400 dollars. Find remainder.',720,'2400-1680.'))
p(110,t('A 140 dollar bag has 5 percent discount. Find discount amount.',7,'140*.05.'),t('Subtract discount from S1 from 140 dollars to find price.',133,'140-7.'))
p(111,t('Apples sell for 4 dollars each. Monday revenue 56 dollars from all picked apples. Find Monday count.',14,'56/4.'),t('S1 is Monday count. Tuesday 12 apples; Wednesday double Tuesday. Find three-day total.',50,'14+12+24.'))
p(112,t('Of 300 bikes sold, half are Trekking, 15 percent BMX, rest MTB. Return counts (Trekking, BMX, MTB).','(150,45,105)','150,45,300-195.'),t('S1 gives counts(Trekking, BMX, MTB). Prices are 450, 250, 500 dollars respectively. Find total revenue.',131250,'67500+11250+52500.'))
p(113,t('Convert 3 hours to minutes.',180,'3*60.'),t('S1 gives picking time in minutes. Rate 2 peaches/minute. Find total peaches.',360,'180*2.'))
p(114,t('A cat runs 15 feet/second and is 5 times as fast as a turtle. Find turtle speed.',3,'15/5.'),t('S1 is turtle feet/second. Find distance in 40 seconds.',120,'3*40.'))
p(115,t('Walter has 60 apples, eats 2/5 of them, then gives away 25 percent of the remainder. Find apples left.',27,'60*.6*.75=27.'),t('S1 gives apples left. Sell all at 3 dollars each. Find revenue.',81,'27*3.'))
p(116,t('A car weighs 1200 pounds, luggage 250 pounds, two children 75 pounds each. Find total weight.',1600,'1200+250+150.'),t('Required push force is 1 percent of weight supplied by S1. Find force in pound-force.',16,'1600*.01.'))
p(117,t('A building has 15 floors and 8 units/floor. Find total units.',120,'15*8.'),t('Three quarters of the units counted in S1 are occupied. Count unoccupied units.',30,'120/4.'))
p(118,t("Rong saves 20 coins/month. Neil saves 2/5 more than Rong, meaning 140 percent of Rong's amount. Find combined monthly saving.",48,'20+28.'),t('S1 is combined coins/month. Starting zero, find total after 10 years.',5760,'48*120.'),note='Explicitly adopts the usual benchmark interpretation of 2/5 times more as 40% more.')
p(119,t('Connection speed 20 kb/second. Convert to kb/hour.',72000,'20*3600.'),t('S1 is kb/hour. 1000 kb=1 Mb in this problem. Convert to Mb/hour.',72,'72000/1000.'))
p(120,t('Isaias sells 3/5 of 300 chickens at 50 dollars each. Find revenue.',9000,'180*50.'),t('S1 is revenue. Desired profit 2000 dollars. Find purchase cost of sold chickens.',7000,'9000-2000.'))
p(121,t('A jar measures 6 by 6 by 15 inches. Jellybeans occupy 80 percent of volume; each bean occupies 0.15 cubic inches. Estimate bean count.',2880,'540*.8/.15.'),t('S1 is estimated total beans; 30 percent are red. Actual red count 900. Find absolute estimation error.',36,'900-.3*2880=36.'))
p(122,t('Jim collects 27 shells, 5 more than Carlos. Carlos collects twice Carrey. Find (Carlos, Carrey).','(22,11)','27-5=22;22/2=11.'),t('S1 gives Carlos and Carrey counts; Jim has 27. Combine and divide equally among 3 people. How many each?',20,'(22+11+27)/3.'))
p(123,t('27000 dollars is divided equally among 3 schools. Find budget per school.',9000,'27000/3.'),t('S1 is budget per school. 100 books cost 500 dollars. How many books per school?',1800,'9000*100/500.'))
p(124,t('Chocolate bar weighs 40 grams, twice each candied apple. Find combined weight of 80 candied apples.',1600,'40/2*80.'),t('S1 is apple weight in grams. Add 25 chocolate bars of 40 grams each. Find total weight.',2600,'1600+1000.'))
p(125,t('Mitchell sold 21 lemonades at 4 dollars each. Find revenue.',84,'21*4.'),t('S1 is Mitchell revenue. Liam revenue 63 dollars. Find excess.',21,'84-63.'))
p(126,t('A bed is 6 feet long, 8 feet wide. Each fabric piece must be 2 feet longer and 2 feet wider. Find area of one piece.',80,'(6+2)*(8+2).'),t('S1 is area of one piece in square feet. Two pieces are needed. Find total area.',160,'2*80.'))
p(127,t('Sum scores 50, 80, 80, 60, 40, 90, 100, 70, 60.',630,'Ninevalues sum630.'),t('S1 is total of nine quiz scores. Find arithmetic mean.',70,'630/9.'))
p(128,t('Joey played 2 matches Monday. Saturday he played double Monday. Find Saturday matches.',4,'2*2.'),t('S1 is Saturday matches. Add 2 Monday and 1 Friday. Find week total.',7,'4+2+1.'))
p(129,t('Howie buys each himself, his teacher, and 25 classmates cupcakes. Count recipients.',27,'1+1+25.'),t('S1 is recipient count. Each gets two cupcakes. Find total.',54,'27*2.'))
p(130,t('A dog digs 6 holes/day for 14 days with none filled. Count initial holes.',84,'6*14.'),t('S1 is initial holes. Each day 9 are filled and 6 new ones appear. At end-of-day accounting, how many whole weeks until backlog reaches zero? Use 7 days/week.',4,'Net3perday;84/3=28days=4weeks.'),note='End-of-cycle convention; within-day temporary absence of holes is not the endpoint.')
p(131,t('Canned tomatoes lose half their volume when cooked. 32 ounces of sauce were produced. Find initial ounces.',64,'32*2.'),t('S1 is initial ounces. Each 16 ounce can contains 3 tomatoes. How many tomatoes?',12,'64/16*3.'))
p(132,t('Two bags of glass contain 72 chips each. 12 chips cover 1 square inch. Find coverable area.',12,'144/12.'),t('S1 is rectangular mosaic area in square inches. Height 3 inches. Find length.',4,'12/3.'))
p(133,t('There are 175 diamonds. Rubies are 35 fewer; emeralds twice rubies. Return (rubies, emeralds).','(140,280)','175-35=140;2*140=280.'),t('S1 gives rubies and emeralds. Add 175 diamonds. Find total gems.',595,'140+280+175.'))
p(134,t('Julia rows 20 feet in 16 seconds at constant rate. How far in 64 seconds?',80,'20*64/16.'),t('S1 is rowing distance in feet. Boat takes 2 liters water per every 10 feet. Find water intake.',16,'80*2/10.'))
p(135,t('Ten sandwiches sell for 7 dollars each. One bread loaf costs 3 dollars and makes 10 sandwiches; cheeses and condiments cost 30 dollars/10 sandwiches. Find (revenue, cost) for 10 sandwiches.','(70,33)','70revenue,3+30cost.'),t('S1 is(revenue, cost)in dollars. Find net profit.',37,'70-33.'))
p(136,t('Using the convention of two masks per outing, Tyrion goes out 3 times/day. Find masks/day.',6,'2*3.'),t('S1 is masks/day. Find masks for 2 days.',12,'6*2.'),note='Original changes twice wording could imply initial+2; benchmark counts2uses/outing.')
p(137,t('Jennifer has 8 puppies, 3 spotted; Brandon 12 puppies, 4 spotted. Return(total, spotted).','(20,7)','8+12,3+4.'),t('S1 gives(total, spotted)puppies. Find percentage spotted.',35,'100*7/20.'))
p(138,t('Feed comprises 2 one-pound boxes, 4 twelve-pound bags, 42 seventy-five-pound bales and 20 sixty-five-pound sacks. Find total weight.',4500,'2+48+3150+1300.'),t('S1 gives the total feed weight in pounds. Each truck trip can carry 2250 pounds. Determine the minimum number of trips and verify that the indivisible items can be allocated. The items are 42 bales of 75 pounds, 20 sacks of 65 pounds, 4 bags of 12 pounds and 2 boxes of 1 pound.',2,'4500/2250=2. First load:17 bales and15 sacks =2250. Remaining25 bales,5 sacks,4 bags,2 boxes also2250.'))
p(139,t('Three women wear 4 inch heels and three wear 2 inch heels. Find mean heel height.',3,'(3*4+3*2)/6.'))
p(140,t('Solve 8+4 m=40 for the number m of mugs.',8,'4m32.'))
p(141,t('Cameron arrives 5 pm and leaves 10 am next day. Beyond first 12 hours, how many additional hours?',5,'17-12=5.'),t('S1 gives additional hours. First 12 hours cost 1000 pesos, additional hours 70 each. 24 hour package cost 1600. Find savings using hourly extension.',250,'1600-(1000+5*70).'))
p(142,t('Charisma walks 5 minutes in each of 8 work hours. Find daily walking minutes.',40,'5*8.'),t('S1 is daily walking minutes. Find total for 5 work days.',200,'40*5.'))
p(143,t('Reading amounts are 4, 20, 7, 8 pages; 15 already read. Find pages remaining.',24,'39-15.'),t('S1 is pages remaining. 4 days left. Find average pages/day needed.',6,'24/4.'))
p(144,t('Monday 32 classes visit, Tuesday twice Monday, Wednesday thrice Monday. Find three-day total.',192,'32+64+96.'),t('S1 gives Mon-Wed total; Thursday 30 and Friday 25. Find weeks total.',247,'192+30+25.'))
defer(145,'indivisible_load_reference_error','80flagstones*75lb=6000lb, but each2000lbtruck fits only26stones.3trucks fit78, so4arerequired. Benchmark3 ignores packing integrality.')
p(146,t('Paisley runs 4 miles; Reggie runs 5 times as far. Find Reggie distance.',20,'4*5.'),t('S1 is Reggie distance, 3 miles farther than Lynn. Find Lynn distance.',17,'20-3.'))
p(147,t('Machine A produces 60 widgets/hour, B produces 45 widgets/hour. Each widget 6 dollars. Find hourly revenue difference.',90,'(60-45)*6.'),t('S1 is hourly difference. Machines run 24 hours/day for 30 days. Find monthly difference.',64800,'90*24*30.'))
p(148,t('An angle increases 5 degrees/hour for 10 hours. Find increase.',50,'5*10.'),t('Initial angle 32 degrees. Add increase from S1.',82,'32+50.'))
p(149,t('40 restaurants receive 2 pounds carrots each. Find used weight.',80,'40*2.'),t('S1 is used weight out of 200 pounds. Find unused weight.',120,'200-80.'))
p(150,t('200 crayons are packed 8 per box. Each empty box 8 ounces. Find total empty box weight.',200,'25boxes*8ounces.'),t('S1 is empty box weight in ounces. 200 crayons weigh 1 ounce each. Find combined weight in pounds at 16 ounces/pound.',25,'(200+200)/16.'))
p(151,t('Of 30 students, 20 percent play football. 25 percent of the rest are cheerleaders or band. Return counts(football, other departures).','(6,6)','6;remaining24*.25=6.'),t('S1 gives disjoint counts of student groups leaving early. Find total departures.',12,'6+6.'))
p(152,t('Leo collected 50 stickers last year and twice as many this year. Find new stickers over those two years.',150,'50+100.'),t('S1 is new stickers; original collection 100. Find current total.',250,'100+150.'))
p(153,t('Naomi caught 17 fish. Darren caught 6 fewer. Find Darren count.',11,'17-6.'),t('Tommy caught 3 more fish than Darren count from S1. Find Tommy count.',14,'11+3.'))
p(154,t('Three people share 1920 dollars. The first gets x, the second x+80, third twice second. Express the combined share as a polynomial in x.','4x+240','x+(x+80)+2(x+80).'),t('S1 gives combined share as a function of first share x. Set it to 1920 and solve for x.',420,'(1920-240)/4.'))
p(155,t('Using elapsed years, calculate years of vacationing from age 23 to age 34.',11,'34-23.'),t('S1 is years. Four vacations/year, one t-shirt block/vacation. Find blocks.',44,'11*4.'),note='Elapsed-year convention, not inclusive counting of birthday years.')
p(156,t('Sheila buys a frying pan 20 percent off 20 dollars and towels 10 percent off 30 dollars. Find new purchase cost.',43,'16+27.'),t('Original card balance 85 dollars; return 15 dollar item; add purchases from S1. Find new balance.',113,'85-15+43.'))
p(157,t('Loyalty reward 1 dollar for every 20 dollars spent. Last trip 80 dollars. Find reward.',4,'80/20.'),t('New bill 43 dollars. Subtract reward S1 and coupon worth twice that reward. Find payment.',31,'43-4-8.'))
p(158,t('Marie buys 12 dollar chicken, 5 milk packs 3 dollars each, 4 apples 1.50 each. Find non pizza cost.',33,'12+15+6.'),t('S1 is non pizza cost. Total paid 50 dollars; pizza box 8.50 each. Find box count.',2,'(50-33)/8.5.'))
p(159,t('A 40 dollar grocery bill has 25 percent vendor fee. Find bill plus fee.',50,'40*1.25.'),t('S1 is bill with vendor fee. Add 3 dollar delivery and 4 dollar tip. Find final total.',57,'50+3+4.'))
p(160,t('Adams mother gives 6 dollars and father twice as much. Find total parent contribution.',18,'6+12.'),t('Trousers cost 30 dollars. S1 gives parent contribution. Find Adams contribution.',12,'30-18.'))
p(161,t('27 cups of raisins are divided equally among cookies, granola, snack mix. Find cookie share in cups.',9,'27/3.'),t('S1 is raisin cups for cookies. Each batch uses 3/4 cup. Find batch count.',12,'9/(3/4).'))
p(162,t('1000 members make 3 posts/day each. Find group posts/day.',3000,'1000*3.'),t('S1 is posts/day. March has 31 days. Find monthly posts.',93000,'3000*31.'))
p(163,t('Claire eats 3 eggs/day for 4 weeks. Find eggs consumed.',84,'3*4*7.'),t('S1 is egg count. Convert to dozens.',7,'84/12.'))
p(164,t('John attends 8 classes/day, 40 minutes/class, 5 days/week. Find weekday learning minutes.',1600,'8*40*5.'),t('S1 is weekday minutes. Each of Saturday and Sunday adds 1/16 of that amount. Find total weekly hours.',30,'1600*(1+2/16)/60.'))
p(165,t('Jon spends 40 min swim, 80 min bike, 50 min run. James swim time is 10 percent less than Jon and bike time 5 min more. Return(Jon total, James nonrun time).','(170,121)','170;36+85=121.'),t('S1 gives(Jon total, James nonrun time)in minutes. James finishes 10 minutes after Jon. Find James runtime.',59,'170+10-121.'),note='Faster interpreted as10percent less elapsedtime, as in benchmark, not speedratio1.1.')
p(166,t('Each sloth berry trip lasts 4 hours. How many complete trips fit 8 hours?',2,'8/4.'),t('S1 is trip count. At least 24 berries needed; equal integer berries/trip. Find minimum per trip.',12,'ceil24/2.'))
p(167,t('Power bank loses 9 percentage points capacity/hour for 5 hours, then 7/hour for 3 hours. Find total percentage points lost.',66,'45+21.'),t('S1 is capacity lost in percentage points. Final charge 28 percent. Find initial charge percent.',94,'66+28.'))
p(168,t('Fabric comes from 11 small shirts 3 sqft each and 8 medium shirts 4 sqft each. Find their combined area.',65,'33+32.'),t('S1 is area of small and medium shirts. Add 6 large shirts 6 sqft each. Find total quilt area.',101,'65+36.'))
p(169,t('Mark works 8 hours/day, 5 days/week. Find weekly hours.',40,'8*5.'),t('S1 is weekly hours. Wage rises from 10 by 2 dollars/hour. Find weekly earnings.',480,'40*12.'))
p(170,t('4 students drink 1.5 quarts and 16 drink 2 quarts each; 10 quarts remain. Find initial quarts.',48,'6+32+10.'),t('S1 is quarts. Use 4 quarts/gallon. Find gallons.',12,'48/4.'))
p(171,t('John runs 3 hours first day and half as long on each of two other days. Find total hours.',6,'3+1.5+1.5.'),t('S1 is total runtime; distance 60 miles. Find average mph.',10,'60/6.'))
p(172,t('A 4000 gallon amount doubles. Find new amount.',8000,'4000*2.'),t('S1 gives current amount. Add 6000 gallons. Find total.',14000,'8000+6000.'))
p(173,t('Tom starts 50 trees, adds 10 and removes 2 each year for 10 years. Find count before death loss.',130,'50+10*(10-2).'),t('30 percent of trees counted in S1 die. Find remaining count.',91,'130*.7.'))
p(174,t('A class has 30 boys, girls 3 times boys, nongendered children 1/10 of boys. Find(girls, nongendered).','(90,3)','30*3,30/10.'),t('S1 gives girls and nongendered counts. Add 30 boys. Find class size.',123,'90+3+30.'))
p(175,t('Five used pens can be recycled into one full pen. Newly made pens can also be used and recycled. Starting with 25 full pens, how many additional pen-uses are possible beyond the original 25?',6,'25empties=>5pens=>5empties=>1pen;1emptystops.'),t('S1 is additional uses beyond 25 original pens. Find total uses.',31,'25+6.'))
p(176,t('Bubbles has 3 puppies, 5 koalas, 2 zebras, 4 frogs. Count current stuffed animals.',14,'3+5+2+4.'),t('S1 gives the number of non-goat stuffed animals. Add g goats so goats are 30 percent of the new total. Solve for g.',6,'g/(14+g)=.3=>.7g4.2.'))
p(177,t('Four schools each send two basketball teams. Find team count.',8,'4*2.'),t('S1 is team count. Each team 5 players and 1 coach. Find total people.',48,'8*6.'))
p(178,t('The pension vests 5 percentage points/year after 20 years of service. Find vested fraction after 30 years.',r'\frac12','(30-20)*.05.'),t('S1 is vested fraction of full 50000 dollar annual pension. Find annual pension.',25000,'50000/2.'))
p(179,t('Two shirts cost 30 dollars each before 40 percent discount. Find undiscounted total.',60,'2*30.'),t('Apply 40 percent discount to total from S1. Find amount paid.',36,'60*.6.'))
p(180,t('Frank gets 5, 15, 30 pairs of shoes. Find total individual shoes added.',100,'2*(5+15+30).'),t('Frank starts 200 individual shoes. Add S1, then remove 180. Find remaining.',120,'200+100-180.'))
p(181,t('Robert 30 cherries, 10 more than Richard. Together with Jerry 60. Find(Richard, Jerry).','(20,10)','Richard20,Jerry60-30-20.'),t('S1 gives(Richard, Jerry). Robert 30. How many more than Jerry?',20,'30-10.'))
p(182,t('Monday 425 letters. Tuesday 17 more than one fifth Monday. Find Tuesday.',102,'425/5+17.'),t('S1 is Tuesday count. Wednesday 5 more than twice Tuesday. Find three-day total including Monday 425.',736,'425+102+(204+5).'))
p(183,t('Shortest tree 6 feet, second 5 feet taller. Find their combined height.',17,'6+(6+5).'),t('Tallest height is twice combined heights from S1. Find height.',34,'2*17.'))
p(184,t('Raymond 6 years older than Samantha, who is 31. Find Raymond age.',37,'31+6.'),t("S1 is Raymond age. He had son at 23. Find the son's age.",14,'37-23.'))
p(185,t('Rubik stages: first 5 minutes, second and third twice that. Find first three total time.',25,'5+10+10.'),t("S1 is first three stages' time. Fourth and fifth take half of 5 minutes each. Completion of fifth finishes whole cube. Find total time.",30,'25+2*2.5.'))
p(186,t('Three people each leave 1/3 of a 12-slice pizza. Find their remaining total slices.',12,'3*12/3.'),t('S1 gives leftovers from three people. Two more people each leave 1/4 of a 12-slice pizza. Find all leftovers.',18,'12+2*3.'))
p(187,t('George has 200 bananas and 45 percent more pears than bananas. Find pears.',290,'200*1.45.'),t('S1 is pears. Add 200 bananas. Find total fruit.',490,'290+200.'))
p(188,t('Work is 3 miles away. Find daily round trip walking distance.',6,'2*3.'),t('S1 is daily miles. Find miles for 5 work days.',30,'6*5.'))
p(189,t('Tatiana has 7 hours Saturday and 5 Sunday. Reads 3 hours, then uses 1/3 of remaining time for games. Find hours for soccer.',6,'12-3=9;9*(2/3)=6.'),t('S1 is soccer hours out of 12 total hours. Find percentage.',50,'100*6/12.'))
p(190,t('John boats 6 hours, swims half of that, watches 3 shows 2 hours each. Find combined hours.',15,'6+3+6.'),t('S1 accounts for 30 percent of vacation time. Sightseeing accounts for 40 percent. Find sightseeing hours.',20,'15/.3*.4.'))
p(191,t('The air conditioner cools 2 degrees/hour for 3 hours and 3 degrees/hour for 4 hours. Find total cooling.',18,'6+12.'),t('S1 is total cooling. Afterwards the room warms 2 degrees/hour for 2 hours. Find net temperature decrease.',14,'18-4.'))
p(192,t('33 voters all vote either for or against. For votes are twice against. Find against votes.',11,'3a=33.'),t('S1 is against votes. For votes are double. Find for votes.',22,'2*11.'))
p(193,t('City sets 15 boxes of 20 fireworks. Hannah sees 40 percent. Find visible city fireworks.',120,'15*20*.4.'),t('S1 is visible city fireworks. Hannah also sees 3 boxes of 5 in her backyard. Find total.',135,'120+15.'))
p(194,t('160 straw pieces distributed: 10 hamsters each 5, rabbits 20, rest to rats. Find rat straw.',90,'160-50-20.'),t('S1 is rat straw. Each rat gets 6 pieces; rats divide equally among 3 cages. Find rats per cage.',5,'90/6/3.'))
p(195,t('Three apples cost 1.50 dollars each, five oranges cost 0.80 dollars each, and six peaches cost 0.75 dollars each. Find total cost in dollars.',13,'4.5+4+4.5.'),t('S1 is cost. Paid 20 dollars. Find change.',7,'20-13.'))
p(196,t('Donations are 500 from neighbors, 1500 online, 200 from friend. Find total excluding lawyer.',2200,'500+1500+200.'),t('Lawyer donates three times the total from S1. Find donation.',6600,'3*2200.'))
p(197,t('Train initial count 120. First stop 20 board. Second stop 50 leave and 100 board. Find count after second stop.',190,'120+20-50+100.'),t('S1 is current train count. Third stop 80 leave, nobody boards. Find final count.',110,'190-80.'),note='120 interpreted as initial total, not120plusRomeo.')
p(198,t('Cars move in the same direction at 60 and 70 mph. Find relative speed.',10,'70-60.'),t('Starting alongside, S1 is separation rate in mph. Find separation after 2 hours.',20,'10*2.'))
p(199,t('Mike 21 points, Jim 3 fewer, Tony twice Mike. Return(Mike, Jim, Tony).','(21,18,42)','21,21-3,2*21.'),t('S1 gives points. Each player with strictly more than 20 gets 1 extra point. Find combined final total.',83,'22+18+43.'),note='Twotimesmore interpreted as twice, consistent with benchmark.')
p(200,t('Pick 30 pounds of berries: 20 dollar entry plus 1.50/pound. Find picking cost.',65,'20+45.'),t('S1 is picking cost. Store price 2.50/pound for 30 pounds. Find savings.',10,'75-65.'))
p(201,t('John has 4 glasses water/day on each of 5 weekdays. Find weekday glasses.',20,'4*5.'),t('S1 is weekday count. On 2 weekend days he replaces one of 4 glasses with soda. Find weekly water glasses.',26,'20+2*3.'))
p(202,t('Of 50 porcupines, 3/5 are female. Each original female has 4 babies/month. Find births/month.',120,'50*3/5*4.'),t('S1 is monthly births. Assume the same original females reproduce for 12 months, no baby maturation and no deaths. Initial population 50. Find final population.',1490,'50+120*12.'),note='Constant reproducing adult cohort assumption explicit; otherwise underspecified.')
p(203,t('Each week 3 cans worth 2 cents each and 5 bottles worth 3 cents each are recycled. Find weekly cents.',21,'6+15.'),t('S1 is weekly cents. Find total for 4 weeks.',84,'21*4.'))
p(204,t('Jill works 2 h Saturday, 1 h Sunday. John works twice her Saturday and thrice her Sunday hours. Find John minus Jill hours.',4,'(4+3)-(2+1).'),t('S1 is hour difference. Both paid 10 dollars/hour. Find earning difference.',40,'4*10.'))
p(205,t('Clara buys 20 pomegranates 20 dollars each and uses 2 dollar voucher. Find payment.',398,'400-2.'),t('S1 is Clara payment. Sheila 20 fruits at 30 dollars each with 10 percent off. Find Sheila payment minus Clara payment.',142,'600*.9-398.'))
p(206,t('Four gallons lemonade each need 1 cup juice. Six lemons yield 1 cup. Find lemons for four regular gallons.',24,'4*6.'),t('S1 is lemons for regular batch. Add one extra gallon at double juice concentration, six lemons/cup. Find total lemons.',36,'24+2*6.'))
p(207,t('Sweater body 900 stitches, collar one tenth of body, rosette twice collar. Find combined non-sleeve stitches.',1170,'900+90+180.'),t('S1 is non-sleeve stitches. Total 1800, two sleeves equal. Find stitches per sleeve.',315,'(1800-1170)/2.'))
p(208,t('Nonfood purchases: bulbs 3 dollars, cups 3, traps 4. Tax 10 percent. Find nonfood total including tax.',11,'(3+3+4)*1.1.'),t('S1 is nonfood taxed cost. Add untaxed milk 2 dollars and eggs 3. Find total.',16,'11+2+3.'))
p(209,t('10 packs contain 30 balloons each. Find initial balloons.',300,'10*30.'),t('S1 is initial count. 12 remain. Find used count.',288,'300-12.'))
p(210,t('Sofie walks 10 miles/day for 7 days. Find total.',70,'10*7.'),t('S1 is Sofie total which is twice Brian total. Find Brian total.',35,'70/2.'))
p(211,t('Cho 14 km/hour for 8 hours, Chloe 9 km/hour for 5 hours. Return their distances.', '(112,45)','14*8,9*5.'),t('S1 gives(Cho, Chloe)distances in km. Find positive difference.',67,'112-45.'))
p(212,t('There are 8 rooms and 4 people. One flashlight/person and two/room. Find flashlights.',20,'4+16.'),t('S1 is flashlights. Four rooms use 4 candles each, four others 5 each. Find combined lights.',56,'20+16+20.'))
p(213,t('Mon, Wed, Fri each have three 1 hour classes. Tue, Thu each have two 2 hour classes. Find weekly class hours.',17,'3*3+2*4.'),t('S1 is weekly hours. Semester 16 weeks. Find semester hours.',272,'17*16.'))

p(214,t('In a series RLC circuit with zero initial conditions, express the total impedance in Laplace variable s. Resistance is R, inductance L, capacitance C.',r'R+Ls+\frac1{Cs}','Series impedances add.'),t('S1 is total series impedance. Output is capacitor voltage, input is voltage across the whole series circuit. Capacitor impedance is 1/(Cs). Find the voltage transfer function.',r'\frac1{LCs^2+RCs+1}','Voltage divider: Zc/Ztotal.'))
p(215,t('A rotor obeys I*domega/dt+2 B*omega=T. With B=.005 and constant T=10, find the steady angular velocity in radians/second.',1000,'Derivativezero;10/(2*.005).'),note='Original question already supplies the differential equation as an earlier solved subproblem.')
p(216,t("Euler method for y'=y^2-x^2 starts (x, y)=(0,-1), step h=.5. Find y after the first step.", '-0.5','-1+.5*1.'),t("S1 gives Euler y at x=.5. For y'=y^2-x^2 and h=.5, find y at x=1.", '-0.5','Slope.25-.25=0.'),t("S2 gives Euler y at x=1. For y'=y^2-x^2 and h=.5, find y at x=1.5.", '-0.875','-.5+.5*(.25-1)=-.875.'))
p(217,t("Light passes from water of refractive index 1.3 toward air of index 1. Use Snell's law to express the critical angle from the water-side normal, in radians.",r'\arcsin(\frac{10}{13})','Set transmitted angle pi/2; sinincident1/1.3.'),note='Equivalent to source np.arcsin(10/13); standard mathematical notation used locally.')
p(218,t('Decompose -(4 s-10)/(s(s+2)(s+5)) as A/s+B/(s+2)+C/(s+5). Return(A, B, C).','(1,-3,2)','Evaluate residues at0,-2,-5.'),t('S1 gives coefficients(A, B, C) of A/s+B/(s+2)+C/(s+5). Find the causal inverse Laplace transform, using unit step u(t).',r'(1-3e^{-2t}+2e^{-5t})u(t)','Inverse transforms of simple poles.'))
p(219,t('Given I(t)=I(0)*exp(-t/(RC)), write it as c*exp(-t/tau). Return c.', 'I(0)','Direct coefficient identification.',kind='forward'))
p(220,t('Bi2 S3 dissolves into 2 Bi^(3+) and 3 S^(2-). Let s be dissolved mol/L. Express Ksp as a monomial in s.', '108s^5','(2s)^2*(3s)^3.'),t('Use Ksp expression from S1 with Ksp=1.6*10^(-72). Find moles dissolved in 3.091*10^6 liters. Return scientific notation with mantissa rounded to 1 decimal place.',r'5.3\times10^{-9}','s=(1.6e-72/108)^(1/5)about1.714e-15; multiply volume.'))
defer(221,'reference_precision_ambiguity','lambda/D with exact radians-to-arcseconds gives0.343775arcsec, not0.33. Supplied approximation convention and grading tolerance need explicit decision.')
p(222,t('Write the complex number 1/(2+2 i) in polar form. Return(magnitude, argument), argument in[-pi, pi].',r'(\frac{\sqrt2}{4},-\frac\pi4)','Magnitude1/sqrt8,argument-pi/4.'),t('S1 gives(A, phi) for 1/(2+2 i). Express Re(exp(it)/(2+2 i)) as a cosine.',r'\frac{\sqrt2}{4}\cos(t-\frac\pi4)','Real part of Aexp(i(t+phi)).'))
p(223,t('A blackbody star has luminosity L=7*10^37 erg/s and temperature T=6*10^5 K. Use L=4*pi*R^2*sigma*T^4 and sigma=5.67*10^-5 cgs. Compute R in cm, to 2 significant figures.',r'8.7\times10^8','sqrt(7e37/(4*pi*5.67e-5*(6e5)^4))=8.706e8.'),note='Uses luminosity already provided in the question; standard constant explicitly supplied, not an inferred subtask answer.')
p(224,t('A runner moves at beta=1/sqrt(2) relative to an emitter. Photons move perpendicular to runner velocity in the emitter frame. Find observed/emitted frequency ratio using gamma(1-beta*cos(theta)), theta=pi/2.',r'\sqrt2','gamma=1/sqrt(1-1/2)=sqrt2.'),t('S1 gives observed/emitted frequency ratio. Emitted wavelength is lambda_red. Find observed wavelength.',r'\frac{\lambda_{\text{red}}}{\sqrt2}','Wavelength inverse to frequency.'))
p(225,t('Use c=3.00*10^8 m/s and frequency 5.09*10^14 Hz. Find wavelength in meters, scientific notation with mantissa 2 decimal places.',r'5.89\times10^{-7}','c/f=5.8939096e-7.'))
p(226,t('Two Boolean inputs each take values 0 or 1. Count distinct input pairs.',4,'2^2.'),t('S1 is the number of input pairs. A Boolean function independently assigns 0 or 1 to each pair. Count such functions.',16,'2^4.'))
p(227,t('An optical star orbital radius is 2.75*10^11 cm; its companion radius is 3*10^12 cm about the same center of mass on opposite sides. Find separation to 2 significant figures.',r'3.3\times10^{12}','Sum3.275e12,round3.3e12.'))
p(228,t("For z'+2 z=exp(2 it), try a particular solution z=A exp(2 it). Determine complex A.",r'\frac1{2+2i}','Substitution(2i+2)A=1.'),t("S1 gives A for a particular solution A exp(2 it) of z'+2 z=exp(2 it). Add the full homogeneous solution, using arbitrary complex C.",r'\frac{e^{2it}}{2+2i}+Ce^{-2t}','Homogeneous z\'+2z=0.'))
p(229,t('In the Bohr hydrogen model v_n=v_1/n with v_1=2.18769*10^6 m/s. Compute v_4, scientific notation with mantissa 2 decimal places.',r'5.47\times10^5','2.18769e6/4=5.469225e5.'),note='Bohr-model constant supplied explicitly; tests calculation rather than recall.')
p(230,t("Compare y''+10 y'+10000 y=0 to y''+2*zeta*omega_n*y'+omega_n^2*y=0. Find positive natural angular frequency omega_n.",100,'sqrt10000.'))
p(231,t('Copper is FCC with 4 atoms per cubic unit cell, molar atomic volume 7.1*10^-6 m^3/mol and Avogadro number 6.02*10^23. Compute lattice parameter a in meters to 3 significant figures.',r'3.61\times10^{-10}','a=(4*7.1e-6/6.02e23)^(1/3).'),t('S1 gives FCC lattice parameter a. Second nearest neighbors lie along cube edges. Return their separation in meters.',r'3.61\times10^{-10}','Second-neighbor distancea.',kind='forward'),note='Material data taken from the repository worked reference; explicitly provided, not independently measured.')
p(232,t('Ignoring order-one constants, use rho=M/R^3 and hydrostatic pressure P=GM^2/R^4. Substitute into P=a*rho^(5/3)-b*rho^(4/3), then multiply by R^5. Return the resulting equation.',r'GM^2R=aM^{5/3}-bM^{4/3}R','Substitute powerlaws and clear R denominators.'),t('Solve S1 for positive radius R in terms of positive G, M, a, b.',r'\frac{aM^{1/3}}{GM^{2/3}+b}','Group Rterms and divide byM^(4/3).'))
defer(233,'factual_not_mathematical','Number of stars in Milky Way is empirical order-of-magnitude recall; no mathematical local reference independently derived here. Omit from this reasoning pilot, record source-stratum loss.')
p(234,t('Convert wavelength 408 nm to angstroms, given 1 angstrom=0.1 nm.',4080,'408/.1.'))
p(235,t('Barium is BCC with 2 atoms per unit cell and molar volume 3.924*10^-5 m^3/mol. Avogadro number 6.02*10^23. Find cubic lattice parameter a in meters to 3 significant figures.',r'5.07\times10^{-10}','(2*3.924e-5/6.02e23)^(1/3)=5.071e-10; source5.08 is a small rounding discrepancy.'),t('S1 is BCC lattice parameter a. Along a<110>face diagonal the repeated segment length a*sqrt(2) contains one effective atom. Find linear density atoms/m, to 3 significant figures.',r'1.39\times10^9','1/(5.07e-10*sqrt2)=1.3947e9.'),note='Material constants from repository worked reference; intermediate lattice parameter recomputed instead of copying5.08.')
p(236,t("A real solution exp(-t/2)*cos(3 t) has characteristic exponents -1/2 plus/minus 3 i. For m*x''+b*x'+k*x=0, use their sum to find b in terms of nonzero m.",'m','Rootsum=-b/m=-1.'))
p(237,t('Population x(t) grows naturally at per-capita rate k and is harvested at constant rate a individuals/year. Write its differential equation.',r'\frac{dx}{dt}=kx-a','Net growth minus harvest.'))
p(238,t('Hydrogen Lyman emission ends at n_f=1. Use 1/lambda=R(1-1/n_i^2), R=1.097*10^7 m^-1, lambda 9.50*10^-8 m. Determine integer upper state n_i consistent with the rounded wavelength.',5,'n_i=sqrt(1/(1-1/(lambda R)))about5; wavelength for5 about9.496e-8.'))
p(239,t('For positive T, simplify the exponent of 10 in sqrt(sqrt((10^(T^2-T))^(1/T))).',r'\frac{T-1}{4}','Divide exponentbyT then2then2.'),t('S1 is exponent of 10. Set T=9 and evaluate 10 raised to it.',100,'Exponent2.'))
p(240,t('Complete squares in p^2-q^2=2 p+6 q+8 and use positive p, q to express p in terms of q.', 'q+4','(p-1)^2=(q+3)^2; bothpositive, p=q+4.'),t('S1 relates p to q. Both are two-digit primes. Find largest p+q.',162,'Largest cousin-prime pair within10..99 is79,83; checks of q>79 give83+4composite,89+4composite,97+4not2digit.'))
defer(241,'unresolved_proof','Minimum Fibonacci difference-basis size needs a valid lower-bound proof; no worked source available. Do not certify formula from benchmark alone.')
p(242,t('For n coins in positions 1 through n, let k be the number of heads and H the sum of their positions. Find the sharp minimum of P=2 H-k^2 for a fixed k. Also show that flipping the kth coin when k>0 reduces P by 1.', 'k','MinimumH=k(k+1)/2 impliesP>=k. Flippingheadatpositionk changesHby-k,kby-1; tail changesHby+k,kby+1; both deltaP=-1.'),t('S1 supplies nonnegative potential reasoning. For independent fair coins, length to all tails equals P=2 H-k^2, H=sum of head positions, k=head count. Calculate its expectation as a function of n.',r'\frac{n(n+1)}4','EH=n(n+1)/4,Ek^2=n/4+n^2/4;2EH-Ek^2=n(n+1)/4. Nonnegativeintegerpotential decreasingby1 proves termination.'))
p(243,t('T=sqrt(26). Factor N=T^20*210^12 into primes. Return the number of distinct prime factors.',5,'26^10*(2*3*5*7)^12; primes2,3,5,7,13.'),t('S1 is the number of distinct primes of N. Count ordered positive pairs(a, b) with ab=N and gcd(a, b)=1.',32,'Each entire primepower must go to exactlyonefactor;2^5.'))
defer(244,'unresolved_geometry','Need a checked derivation of four subtriangle inradii and validity of the intersection configuration; final14 alone is not local ground truth.')
defer(245,'unresolved_functional_equation','Candidate f(x)=x-1 verifies, but uniqueness over all real functions has not been established here. Substitution alone must not certify all solutions.')
p(246,t('April 1, 2012 was Sunday. Using Gregorian leap years, find the first subsequent year in which April 1 falls on Sunday.',2018,'Shifts1,1,1,2,1,1 cumulatively7by2018.'))
p(247,t('Angle A is interior to a triangle and tan(A)=-4/3. Find sin(A).',r'\frac45','Aobtuse;sinpositive,3-4-5ratio.'),t('S1 is sin(A). Two adjacent sides have lengths 5 and 6. Find triangle area.',12,'1/2*5*6*4/5.'))
p(248,t('Let C be current sum of children ages and k their number. Parents current age sum 6 C. Two years ago parent sum 4 less, child sum 2 k less, with ratio 10. Derive C in terms of k.','5k-1','6C-4=10(C-2k)=>4C20k-4.'),t('Use C from S1. Six years later parents sum 6 C+12, children sum C+6 k, ratio 3. Solve for k.',3,'6C+12=3C+18k=>C=6k-4;5k-1=6k-4.'))
p(249,t('A cube of edge 37 has a centered 1-by-1 opening on each of its 6 faces. Find the remaining original exterior area.',8208,'6*37^2-6.'),t('S1 is remaining original exterior area. Three perpendicular centered unit-square tunnels intersect in a unit cube. Each tunnel has four wall strips with total exposed length 37-1. Find total exterior plus tunnel-wall area.',8640,'8208+12*36.'))
p(250,t('100 marbles have black: gold ratio 1: 4. Find (black, gold).','(20,80)','5parts20each.'),t('S1 gives(black, gold). Add only gold so new ratio 1: 6. How many gold added?',40,'20*6-80.'))
p(251,t('For each element of a 6-element set, consider membership bits in subsets(A, B, C). Count per-element assignments satisfying A subset C and B subset C.',5,'IfC0only00; ifC1any4assignments.'),t('S1 is per-element pattern count for A, B subsets C; reversed inclusion has same count. On 6 elements count triples satisfying either condition. Both conditions hold iff A=B=C.',31186,'2*5^6-2^6=31250-64.'))
p(252,t('In a uniform random permutation of 1 through 7, find expected value at any specified position.',4,'(1+...+7)/7=4.'),t('S1 is mean at any position. Find expected a1-a2+a3-a4+a5-a6+a7.',4,'Linearity;4positive-3negativecopies.',kind='synthesize'))
p(253,t('A positive integer is fibbish when every digit after its first two is at least the sum of the previous two. Determine the maximum possible digit length.',8,'Smallest initialpair(1,0) followedbyminimumdigits1,1,2,3,5,8 allows8; ninthmin13impossible. Any bigger prefix cannot increase length.'),t('S1 is maximum fibbish length. Find lexicographically largest positive decimal digit sequence of that length satisfying d_i>=d_(i-1)+d_(i-2). Return the integer.',10112369,'Backward feasibility:1,0,1,1,2,3 forcedfor8length; seventh<=6 as eighth>=3+seventh<=9. Choose6then9. DFScheckalso.'))
p(254,t('Three mixtures: 40 g at 10 percent acid, 50 g at 20 percent, 50 g at 30 percent. Find(total mass, total acid mass) in grams.','(140,29)','40+50+50;4+10+15.'),t('S1 gives total mass and acid. Remove 60 g of 25 percent acid mixture. Find acid percentage in remaining mixture.','17.5','Remaining80g,acid29-15=14;100*14/80.'))
p(255,t('A flight was planned to last 130 minutes but takes 100 minutes. Its actual speed is 1.5 miles/minute faster than planned. Determine the flight distance in miles.',650,'D*(30/13000)=1.5.'))
p(256,t('List exponents of all positive divisors of 2^20 as an arithmetic progression; compute their sum.',210,'0+1+...+20=20*21/2.'))
p(257,t('Find the inclusive interval of integer bases b>=2 in which 41 has exactly two digits. Return the endpoint pair.','(7,41)','b<=41<b^2.'),t('S1 is an inclusive interval of integer bases. Count its elements.',35,'41-7+1.'))
defer(258,'unresolved_proof','Maximal colors for matching slice color-sets needs construction and upper bound; not verified from the final formula alone.')
defer(259,'unresolved_geometry','Excircle perpendicular-foot construction requires checking position and angle degeneracies; no verified local references yet.')
p(260,t('Count positive even integers n for which n!! divides 2012!!, where n!!=n(n-2)...2.',1006,'All evenn<=2012divide; largerdoublefactorialexceeds target.'),t('Count positive odd integers n for which n!! divides 2012!!. Use prime valuations or exact divisibility, and justify the first failing value. Return only the count in the result.',504,'Exact integer productcheck shows everyodd1..1007divides;1009isprime notin2012!!=2^1006*1006!, so everyodd>=1009fails.'),t('S1 and S2 count qualifying even and odd positive integers. Find total.',1510,'1006+504.'))
p(261,t('Triangle ABC lengths AB=4, BC=6, AC=8. Find cos(angle ABC).',r'-\frac14','(16+36-64)/(2*4*6).'),t('Using S1, place B=(0, 0), A=(4, 0), C above x-axis with BC=6. Squares ABQR and BCST are external to triangle. Find QT.',r'2\sqrt{10}','C=(-3/2,3sqrt15/2);Q=(0,-4),T=(-3sqrt15/2,-3/2); QT^2=135/4+25/4=40.'))
p(262,t('Enumerate four-digit positive integers with digit-square sum 17. Return(count, lower_middle, upper_middle) in numerical order; repeat the middle if count is odd.','(15,2302,2302)','Exact enumeration gives15 values; eighth is2302.'),t('S1 gives count and central value(s). Find the median.',2302,'Odd count; repeated central value2302.'))
p(263,t('Find the largest 5-digit multiple of 63 with five distinct decimal digits. Justify that no larger candidate qualifies.',98721,'Descendingmultiples:99981,99918,99855,99792,99729,99666,99603,99540,99477,99414,99351,99288,99225,99162,99099,99036,98973,98910,98847,98784,98721; onlylastfirstwithdistinctdigits.'))
p(264,t('For x in[0, 360]degrees with sin, cos, tan, cot all finite, consider sin(x)=cot(x) or cos(x)=tan(x). Find the sum of all x satisfying either equality, excluding any with fewer than 3 distinct values among the four functions.',540,'Firstcosx=(sqrt5-1)/2 givespairangles summing360; secondsinx=same givespair summing180. Allhaveexactly3values.'),t('For x in[0, 360]degrees, tan(x)=cot(x) with exactly 3 distinct finite values among sin, cos, tan, cot. Find the sum of such x.',450,'Candidates45,135,225,315;45and225only2values;135+315=450.'),t('S1 and S2 sum disjoint solutions for the possible equalities among sin, cos, tan, cot. Other equalities either undefined or give only 2 distinct values. Find sum of all x yielding exactly 3 distinct values.',990,'540+450.'))
p(265,t('For a closed cylinder, volume V=pi*r^2*h and total area A=2 pi*r*h+2 pi*r^2. Simplify A/(2 V).',r'\frac1r+\frac1h','Dividebothterms.'),t('S1 expresses A/(2 V). Given V=1 and A=12, find 1/r+1/h.',6,'12/(2*1).'))
p(266,t('An arithmetic progression begins 1 and ends 19. Find mean of all terms.',10,'(1+19)/2.'),t('S1 is mean and total sum 70. Find term count.',7,'70/10.'))
defer(267,'malformed_reference','Reference fraction has missing numerator and d in denominator. Cyclelength conventions (especially k=3) also need resolution; no silent restoration.')
p(268,t('Compute floor((1+sqrt(8*2011-7))/2).',63,'126^2<16081<127^2, so floor is63.'),t('S1 is the floor term. Compute g(2011)=2*2011+S1.',4085,'4022+63.'))
p(269,t('A rectangle[0, m]x[0, 1]has integer m>0. Triangulate with area 1/2 triangles whose vertices are boundary lattice points. By advancing a top/bottom frontier, derive the count in terms of m.',r'\binom{2m}{m}','Everytriangle uses one unit horizontal base; frontier advances one of two rows. Choose order of mtop andmbottom advances; bijective.'),t('Evaluate count formula from S1 at m=7.',3432,'C14,7=3432.'))
p(270,t('Set t=log 10(x), k=floor(t^2), with t=(3+k)/4. Find all allowable nonnegative integers k.',r'\{0,1,9,10\}','k<=((k+3)/4)^2<k+1; solve the two quadratic inequalities.'),t('S1 gives k values. Corresponding x=10^((k+3)/4). Their product equals 10^y. Find y.',8,'Sum exponents(3+4+12+13)/4=8.'))
p(271,t('Three lawnmowers take 48, 48, 24 minutes individually per lawn. Find combined lawns/minute.',r'\frac1{12}','1/48+1/48+1/24.'),t('S1 is combined rate. Find lawns in 2 hours.',10,'120/12.'))
p(272,t('Cross-multiply (x^2+x+4)/(2 x+1)=4/x, for x not 0,-1/2. Factor the resulting polynomial.','(x+1)(x-2)(x+2)','x^3+x^2-4x-4.'),t('Use S1 to find all roots except excluded x=0,-1/2.',r'\{-2,-1,2\}','All three roots satisfy the denominator restrictions.'))
p(273,t('T=72 sqrt(2), K=(T/12)^2. Compute K.',72,'(6sqrt2)^2.'),t('S1 is K. Find sum of positive magnitudes j/2 for j=1,..., K.',1314,'K(K+1)/4.'),t('S1 is K. Find sum of j/2 over j divisible by 3 in 1,..., K.',450,'(3/2)*24*25/2.'),t('S2 is sum of all magnitudes; S3 sum of those whose sign must be negative. Find signed total.',414,'1314-2*450.'))
p(274,t('Inside the unit cube, find volume of points within taxicab distance 3/5 of vertex(0, 0, 0).',r'\frac9{250}','Tetrahedronlegs3/5,volume(3/5)^3/6.'),t('Find overlap volume of taxicab balls radius 3/5 about adjacent vertices(0, 0, 0),(1, 0, 0), restricted to unit cube.',r'\frac1{3000}','Letu=y+z<.1; x-range.2-2u,volumeintegral0...1 u(.2-2u)du=1/3000.'),t('S1 is removed volume per vertex, S2 overlap per edge. Unit cube has 8 vertices 12 edges. No other overlaps or triple overlaps occur. Find remaining volume.',r'\frac{179}{250}','1-8*9/250+12/3000.'))
p(275,t('Choose two distinct tickets uniformly from 1 through 6. Count pairs whose minimum is greater than 4.',1,'Only{5,6}.'),t('S1 is unfavorable count among C(6, 2)pairs. Find probability minimum at most 4.',r'\frac{14}{15}','1-1/15.'))
defer(276,'unresolved_combinatorial_proof','Napkin-overlap extremum needs a checked construction and matching upper bound, not only the provided integer3986729.')
p(277,t('Rectangular prism face areas ab 27, ac 32, volume abc 144. Find bc.',24,'(abc)^2/(ab*ac)=20736/864.'),t('S1 is third face area; other two are 27 and 32. Find total surface area.',166,'2*(27+32+24).'))
p(278,t('K=9*7. Find the largest perfect square m<=K and its positive square root. Return(m, sqrt(m)).','(49,7)','49<=63<64.'),t('The sequence resets to A_m=sqrt(m) at perfect-square indices and otherwise increases by 1. S1 gives the last perfect square at or below 63 and the reset value there. Find A_63.',21,'7+(63-49).'))
p(279,t('Given sec(x)-tan(x)=3, find sec(x)+tan(x).',r'\frac13','Product is1.'),t('S1 is sec+tan; sec-tan=3. Find sin(x).',r'-\frac45','sec5/3,tan-4/3,ratio-4/5.'))
p(280,t('Digits A in 1..9 and B, C in 0..9 satisfy(10 A+B)^2+(10 A+C)^2=1313. Find(A,{B, C}).',r'(2,\{2,9\})','Only22^2+29^2=1313, allowing interchange.'),t('S1 gives A and the other two digits. Find A+B+C.',13,'2+2+9.'))
p(281,t('Five consecutive integers start at n. The first three squares sum to the last two squares. Derive a monic quadratic equation for n.','n^2-8n-20=0','Expand and collect.'),t('Solve S1 and return the two increasing five-tuples.',r'\{(-2,-1,0,1,2),(10,11,12,13,14)\}','Roots-2,10.'))
defer(282,'reference_symmetry_and_proof','Listedtriples omit x/y permutations despite orderedvariable statement. Need proof of exhaustiveness and an explicit unordered-pair grading convention before local references are certified.')
p(283,t('An Eden sequence is nonempty, increasing, drawn from 1.. N, with odd values in odd positions and even values in even positions. Let e(N)count them. Derive e(N)in terms of e(N-1), e(N-2) for N>=3.', 'e(N-1)+e(N-2)+1','SequencesnotendingNcount e(N-1). EndingN correspond alternating subsequences endingoppositeparity; incrementdN=dN-1+dN-2 implies eN=eN-1+eN-2+1, bases1,2.'),t('S1 gives Eden recurrence; e17=4180, e20=17710. Find(e18, e19).','(6764,10945)','Letu=e18,e19=u+4181,e20=2u+4182;u6764.'))
p(284,t('Initially isotope A has twice as many atoms as B. A has a 6-minute half-life. After 24 minutes the two isotopes have equal atom counts. Find the fraction of B remaining relative to its initial count.',r'\frac18','Aremaining2*B0/16=B0/8.'),t('S1 is fraction of B remaining after 24 minutes. Find constant B half-life in minutes.',8,'1/8=2^-3;24/3.'))
p(285,t('Positive integer cent prices a, p obey 500<5 a+5 p<1000 and 1000<2 a+12 p<2000. Find the least feasible p.',61,'a+p<=199 so2a+12p<=398+10p>1000impliesp>=61.Witnessa138,p61gives995and1008.'),t('S1 is minimum peach price in cents. Find cost of 25 peaches.',1525,'25*61.'))
defer(286,'unresolved_topological_proof','Labyrinth invariant and sharpness for all line arrangements not verified here; cannot certify n+1 from answer alone.')
defer(287,'unresolved_extremal_proof','At-most-k local distance-scale theorem needs checked upper bound for arbitrary real sets; no proof yet, not a model-performance exclusion.')
p(288,t('A chord of length 6 in a radius-5 circle has minor central angle alpha. Find cos(alpha).',r'\frac7{25}','sin(alpha/2)=3/5,cosalpha=1-18/25.'),t('A cyclic hexagon has four consecutive-arc chords of 6 and two equal other chords. Its radius 5; S1 is cos alpha for the 6 chords. Each remaining central angle beta=pi-2 alpha. Find remaining chord length.',r'\frac{14}{5}','10sin(beta/2)=10cosalpha=14/5. Positivecyclicarcconditionsruleoutmajorarcchoices.'))
p(289,t('Let L0(t)=product(1-t/j) for j=1..n. Find the coefficient of x^n*y^n*z^n in L0(x)L0(y)L0(z).',r'\frac{(-1)^{3n}}{(n!)^3}','Eachleadingcoefficient(-1)^n/n!.'),t('Let L0(t)=product(1-t/j) for j=1,..., n. S1 gives the coefficient of x^n*y^n*z^n in L0(x)L0(y)L0(z). For the grid {0,..., n}^3, find the minimum number of planes avoiding the origin that cover all other grid points. Prove a lower bound using polynomial interpolation and supply a construction.', '3n','Productofplaneequationshasnonzerovalueatoriginandzeroatothergridpoints.Ifdegree<3n,multivariateLagrangetopcoefficientmustbe0butindicatorcoefficientfromS1timesnonzerovalueisnot0.Constructionx+y+z=j,j1..3n.'))
p(290,t('A function on positive integers satisfies f(1)=1, f(n)=f(n/2) for even n, and f(n)=f(n-1)+1 for odd n>1. Calculate f(50).',3,'f50=f25=f24+1=f12+1=f6+1=f3+1=f2+2=f1+2=3.'))
p(291,t('Factor 15!-13!as 13!times an integer and factor that integer into primes.', '13!*11*19','15*14-1=209=11*19.'),t('S1 factors 15!-13!. Find its largest prime divisor.',19,'13!hasnoprimeabove13;additional19.'))
p(292,t('f(x)+g(x)=3 x+5, f(x)-g(x)=5 x+7. Find(f(2), g(2)).','(14,-3)','Sum11,difference17;f14,g-3.'),t('S1 gives(f(2), g(2)). Compute 2*f(2)*g(2).',-84,'2*14*(-3).'))
defer(293,'unresolved_polynomial_proof','Minimal-degree polynomial recognizing unordered regular polygons needs both construction and lower bound. Not established in this self-review.')
defer(294,'unresolved_sharp_inequality','Sharp constant4/9 requires a finite-n inequality proof and sharpness sequence. Candidate benchmark alone is not sufficient.')
p(295,t('Find least integer strictly greater than 2023 with digit sum 17.',2069,'20abrequiresa+b15; smallesttens6,units9; all2024..2059digitsums<17.'))
p(296,t('Two dice have faces 2, 3, 5, 7, 11, 13. Count ordered outcomes with prime sum.',6,'Sumprimeodd requiresone2; other3,5,11yield5,7,13. Twoordersfor3choices.'),t('S1 is favorable count of 36 equiprobable outcomes. Find probability.',r'\frac16','6/36.'))
defer(297,'unresolved_tiling_proof','Exact-one-covered-neighbor domino extremum needs verified construction and bound for every n; not certified by the formula alone.')
p(298,t('Nine lights are initially off. Each minute exactly five different switches are flipped. Find the minimum number of minutes needed to make all lights on. Justify a lower bound and give a matching sequence of flips.',3,'1minuteinsufficient;2minutes10flips cannotgive9oddcounts(parity).3roundsflip{1,2,3,4,5},{1,2,3,6,7},{1,2,3,8,9}.'))
p(299,t('log_a(c)=log_b(d)=T nonzero, with a, b, c, d positive, a, b not 1 and ab not 1. Express log_(sqrt(ab))((cd)^3)in terms of T.', '6T','cd=(ab)^T;baselogdivision(3T)/(1/2).'),t('S1 is numerator in T. Divide by T+T for T nonzero.',3,'6T/(2T).'),note='ab!=1 is a necessary domain condition for the expression; original omitted it.')
p(300,t('Factor ab(a^2-b^2)+bc(b^2-c^2)+ca(c^2-a^2) into linear factors.','-(a-b)(b-c)(c-a)(a+b+c)','Expansion verifies sign.'),t('Let u+v+w=0 and u^2+v^2+w^2=R^2. Find maximum |(u-v)(v-w)(w-u)| as a function of R>=0.',r'\frac{R^3}{\sqrt2}','Cubic discriminant R^6/2-27(uvw)^2; maximum attained with one zero and other two opposite.'),t('Use S1 and S2. Put s=a+b+c and centered coordinates u=a-s/3, v=b-s/3, w=c-s/3. Squared norm is s^2/3+R^2. Maximize |s|R^3/sqrt(2)/(s^2/3+R^2)^2 to obtain the sharp constant.',r'\frac{9\sqrt2}{32}','For R>0 let x=|s|/R; derivative gives maximum at x=1,value9/(16sqrt2). R=0 gives0. Equality is attainable.'))


NOTATION = {
    86: 'C_n^k is the binomial coefficient; use explicit binom notation, not a variable C with powers.',
    217: 'np.arcsin(10/13) denotes inverse sine in radians; use mathematical arcsin notation.',
    224: 'Remove the text command from the red subscript; the variable still denotes emitted red wavelength.',
    281: 'Source concatenates two consecutive five-tuples. Preserve both solutions, explicitly grouped as tuples.',
}
for _i in (220, 223, 225, 227, 229, 231, 235):
    NOTATION[_i] = ('Source e-notation denotes decimal scientific notation, not Euler number multiplication. '
                    'Write the same number as mantissa times 10 to an integer power.')


def build():
    from check_math import run
    exact_checks = run(PLANS)
    assert set(PLANS) | set(ISSUES) == set(range(1, 301))
    assert not set(PLANS) & set(ISSUES)
    source = ROOT / 'experiments/worker-specialization-300-v1/review/selection.json'
    selection = evaluation.read_json(source)
    assert selection['selection_sha256'] == 'c5baddf888a22c799519b83aae2cbafd2cbf7f7d86ccb1e172af7cbb19fa853e'
    rows = []
    for number, original in enumerate(selection['splits']['reserved'], 1):
        included = number in PLANS
        row = dict(original, question_number=number,
            decision='include' if included else 'exclude',
            author=AUTHOR, checker=AUTHOR, author_type='model', checker_type='model',
            checks={key: included for key in study.CHECKS},
            tasks=PLANS.get(number, []),
            exclusion_reason='' if included else ISSUES[number]['reason'],
            exclusion_category='' if included else ISSUES[number]['category'],
            review_note=NOTES.get(number, ''),
            computational_checks=[c for c in exact_checks if c['question_number'] == number])
        if number in NOTATION:
            row['canonical_final_answer'] = PLANS[number][-1]['reference_answers'][0]
            row['final_answer_notation_note'] = NOTATION[number]
        rows.append(row)
    review = dict(selection_sha256=selection['selection_sha256'], split='reserved',
        review_mode='model_self_review', purpose='exploratory_model_review',
        independently_reviewed=False, worker_outputs_inspected=False,
        evidence_note='Model authored and self-reviewed before reserved worker inference. '
            'Partial computational checks do not certify every proof. Previously seen development/log examples '
            'and the historical validation source mean this is not semantically unseen test data.',
        examples=rows)
    study.validate_review(selection, review, exploratory_self_review=True)
    out = Path(__file__).resolve().parent
    evaluation.write_json(out / 'review.json', review)
    summary = dict(sampled_questions=len(rows), included_questions=len(PLANS),
        deferred_questions=len(ISSUES), independently_reviewed=False,
        tasks=sum(map(len, PLANS.values())),
        prompts_per_model=sum(2*len(tasks)-1 for tasks in PLANS.values()),
        task_count_distribution=dict(sorted(Counter(map(len, PLANS.values())).items())),
        computational_checks=len(exact_checks),
        questions_with_computational_checks=len({c['question_number'] for c in exact_checks}),
        by_stratum={name: dict(sampled=sum(study.stratum(r)==name for r in rows),
            included=sum(study.stratum(r)==name and r['decision']=='include' for r in rows))
            for name in sorted({study.stratum(r) for r in rows})},
        exclusions=[dict(number=r['question_number'], question_id=r['question_id'],
            stratum=study.stratum(r), category=r['exclusion_category'], reason=r['exclusion_reason'])
            for r in rows if r['decision']=='exclude'],
        normalization_count=len(NOTATION),
        reference_checks=exact_checks)
    evaluation.write_json(out / 'review-summary.json', summary)
    study.freeze(source, out / 'review.json', out / 'frozen', exploratory_self_review=True)
    lines = ['EXPLORATORY MODEL SELF-REVIEW; NOT INDEPENDENT VALIDATION',
             'All 300 sampled questions retained in this review; exclusions are explicit.', '']
    for row in rows:
        lines += [f"QUESTION {row['question_number']} [{study.stratum(row)}] {row['decision'].upper()}",
            row['question'], 'Original final reference: ' + row['final_answer']]
        if row.get('canonical_final_answer'):
            lines += ['Canonical final reference: ' + row['canonical_final_answer'],
                      'Notation note: ' + row['final_answer_notation_note']]
        if row['review_note']:
            lines += ['Review note: ' + row['review_note']]
        for index, task in enumerate(row['tasks'], 1):
            lines += [f"S{index} ({task['kind']}): {task['instruction']}",
                'Local reference: ' + task['reference_answers'][0],
                'Math check: ' + task['reference_check']]
        if row['exclusion_reason']:
            lines += ['Deferred: ' + row['exclusion_reason']]
        lines += ['', '='*80, '']
    with (out / 'review-cards.txt').open('x') as stream:
        stream.write('\n'.join(line.rstrip() for line in '\n'.join(lines).splitlines()) + '\n')
    print(json.dumps({key: value for key, value in summary.items()
                     if key not in {'exclusions', 'reference_checks'}}, indent=2))


if __name__ == '__main__':
    build()
