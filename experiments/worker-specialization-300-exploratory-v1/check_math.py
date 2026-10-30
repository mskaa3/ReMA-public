"""Exact arithmetic/enumeration checks, not an independent reviewer or proof checker."""
import ast
from fractions import Fraction as F
import itertools
import math
import re


def number(s):
    s = s.strip()
    if s.startswith('('):
        return tuple(F(str(v)) for v in ast.literal_eval(s))
    match = re.fullmatch(r'\\frac\{?(\d+)\}?\{?(\d+)\}?', s)
    if match:
        return F(int(match[1]), int(match[2]))
    return F(s)


def run(plans):
    checks = []
    def verify(i, stage, expected, description):
        actual = number(plans[i][stage - 1]['reference_answers'][0])
        assert actual == expected, (i, stage, actual, expected)
        checks.append(dict(question_number=i, stage=stage, check=description))
    # Recalculate each grade-school task from its stated quantities, independently of labels.
    cases = {
        92:(F(9,2),8),93:(17+7,(17+7)*F(2,3)),94:(2+1,2*(2+1)),
        95:((85-25-12)//3,(85-25-12)//3+12),96:(2*150//75,2*150//75+6),
        97:(5000-3755,(5000-3755)//15),98:((25+35)*F(7,10),50-(25+35)*F(7,10)),
        99:(52+5,F(47+52+57,3)),100:(6*20,6*20+67),101:(5-3,6*(5-3)),
        102:(18+21+15,72-18-21-15),103:(10*15*F(3,5),10*15*F(3,5)-40),
        104:(10*25000*F(7,10),10*25000*F(7,10)*F(1,10)),105:(5-3,5+5-3),
        106:(300*F(1,2)+4*30,200+300*F(1,2)+4*30+2*6),107:(5*25+2*10,5*25+2*10-55),
        108:(65//13,65//13*12+15),109:(2400*F(7,10),2400*F(3,10)),
        110:(140*F(5,100),140*F(95,100)),111:(56//4,56//4+12+2*12),
        112:((150,45,105),150*450+45*250+105*500),113:(3*60,3*60*2),
        114:(15//5,15//5*40),115:(60*F(3,5)*F(3,4),60*F(3,5)*F(3,4)*3),
        116:(1200+250+2*75,(1200+250+2*75)*F(1,100)),117:(15*8,15*8//4),
        118:(20+20*F(7,5),(20+20*F(7,5))*120),119:(20*3600,20*3600//1000),
        120:(300*F(3,5)*50,300*F(3,5)*50-2000),
        121:(6*6*15*F(4,5)/F(15,100),900-6*6*15*F(4,5)/F(15,100)*F(3,10)),
        122:((27-5,(27-5)//2),(27+22+11)//3),123:(27000//3,27000//3*100//500),
        124:(80*40//2,80*40//2+25*40),125:(21*4,21*4-63),
        126:((6+2)*(8+2),2*(6+2)*(8+2)),127:(sum([50,80,80,60,40,90,100,70,60]),630//9),
        128:(2*2,2+1+2*2),129:(1+1+25,2*(1+1+25)),130:(6*14,(6*14)//(9-6)//7),
        131:(32*2,32*2//16*3),132:(2*72//12,2*72//12//3),
        133:((175-35,2*(175-35)),175+3*(175-35)),134:(20*64//16,20*64//16//10*2),
        135:((10*7,3+30),10*7-3-30),136:(2*3,2*3*2),137:((8+12,3+4),100*F(7,20)),
        138:(2+4*12+42*75+20*65,2),139:(F(3*4+3*2,6),),140:((40-8)//4,),
        141:(7+10-12,1600-(1000+(7+10-12)*70)),142:(8*5,8*5*5),
        143:(4+20+7+8-15,(4+20+7+8-15)//4),144:(32*(1+2+3),32*6+30+25),
        146:(4*5,4*5-3),147:((60-45)*6,(60-45)*6*24*30),148:(5*10,32+5*10),
        149:(40*2,200-40*2),150:(200//8*8,(200//8*8+200)//16),
        151:((30*F(1,5),30*F(4,5)*F(1,4)),30*F(1,5)+30*F(4,5)*F(1,4)),
        152:(50+2*50,100+50+2*50),153:(17-6,17-6+3),154:(None,(1920-3*80)//4),
        155:(34-23,(34-23)*4),156:(20*F(4,5)+30*F(9,10),85-15+20*F(4,5)+30*F(9,10)),
        157:(80//20,43-3*(80//20)),158:(12+5*3+4*F(3,2),F(50-33)/F(17,2)),
        159:(40*F(5,4),40*F(5,4)+3+4),160:(6+2*6,30-6-12),161:(27//3,F(27//3)/F(3,4)),
        162:(1000*3,1000*3*31),163:(3*4*7,3*4*7//12),164:(8*40*5,F(8*40*5)*F(9,8)/60),
        165:((40+80+50,40*F(9,10)+80+5),40+80+50+10-(40*F(9,10)+80+5)),
        166:(8//4,24//(8//4)),167:(9*5+7*3,9*5+7*3+28),168:(11*3+8*4,11*3+8*4+6*6),
        169:(8*5,8*5*(10+2)),170:(4*F(3,2)+16*2+10,(4*F(3,2)+16*2+10)/4),
        171:(3+2*F(3,2),F(60)/(3+2*F(3,2))),172:(4000*2,4000*2+6000),
        173:(50+10*(10-2),(50+10*(10-2))*F(7,10)),174:((30*3,30//10),30+30*3+30//10),
        175:(25//5+5//5,25+25//5+5//5),176:(3+5+2+4,F(14)*F(3,7)),
        177:(4*2,4*2*6),178:(F(30-20,20),50000*F(30-20,20)),179:(2*30,2*30*F(3,5)),
        180:(2*(5+15+30),200+2*(5+15+30)-180),181:((30-10,60-30-(30-10)),30-(60-30-(30-10))),
        182:(425//5+17,425+(425//5+17)+2*(425//5+17)+5),183:(6+(6+5),2*(6+6+5)),
        184:(31+6,31+6-23),185:(5+10+10,5+10+10+2*F(5,2)),186:(3*12//3,3*12//3+2*12//4),
        187:(200*F(145,100),200+200*F(145,100)),188:(2*3,2*3*5),
        189:((7+5-3)*F(2,3),100*F(6,12)),190:(6+3+3*2,F(15)/F(3,10)*F(4,10)),
        191:(2*3+3*4,2*3+3*4-2*2),192:(33//3,33//3*2),193:(15*20*F(2,5),15*20*F(2,5)+3*5),
        194:(160-10*5-20,(160-10*5-20)//6//3),195:(3*F(3,2)+5*F(4,5)+6*F(3,4),20-13),
        196:(500+1500+200,3*(500+1500+200)),197:(120+20-50+100,120+20-50+100-80),
        198:(70-60,(70-60)*2),199:((21,21-3,2*21),22+18+43),200:(20+30*F(3,2),30*F(5,2)-(20+30*F(3,2))),
        201:(4*5,4*5+2*3),202:(50*F(3,5)*4,50+50*F(3,5)*4*12),203:(3*2+5*3,(3*2+5*3)*4),
        204:(2*2+3*1-2-1,(2*2+3*1-2-1)*10),205:(20*20-2,20*30*F(9,10)-(20*20-2)),
        206:(4*6,4*6+2*6),207:(900+90+180,(1800-900-90-180)//2),208:((3+3+4)*F(11,10),5+10*F(11,10)),
        209:(10*30,10*30-12),210:(7*10,7*10//2),211:((14*8,9*5),14*8-9*5),
        212:(4+8*2,4+8*2+4*4+4*5),213:(3*3+2*2*2,(3*3+2*2*2)*16),
    }
    for i, values in cases.items():
        for j, value in enumerate(values,1):
            if value is not None:
                verify(i,j,value,'Exact rational calculation from stated quantities')
    assert 17*75+15*65 == 2250 == 25*75+5*65+4*12+2
    checks.append(dict(question_number=138,check='Explicit two-load packing witness'))
    dp=[0]*1001
    successful=[]
    for n in range(1,1001):
        dp[n]=1+min(dp[n-c] for c in (1,10,25) if n>=c)
        q,r=divmod(n,25)
        good=q+r//10+r%10==dp[n]
        assert good == (q==0 or r%10<5)
        successful.append(good)
    verify(50,1,sum(r%10<5 for r in range(25)),'All residue classes checked against DP through1000')
    verify(50,2,sum(successful),'Dynamic programming optimal coin count, every N=1..1000')
    exact = {
      (2,2):81,(3,2):10,(6,1):3**2*8,(6,2):4*6**3//3//72,
      (9,1):99**2,(9,2):99**2+99+1,(14,1):8,(15,1):4,(15,2):10,
      (18,1):15**2*24,(18,2):15**2*24//9,(23,1):F(2,3)**3,(23,2):2*F(2,3)**3,
      (25,2):sum(a*a+b*b<=25 for a in range(-5,6) for b in range(-5,6)),
      (34,1):sum((a+b)%2==0 for a in (2,3,5,7,11,13) for b in (2,3,5,7,11,13)),
      (49,1):sum(math.comb(4,j)*math.comb(6,4-j) for j in (2,3,4)),
      (69,1):max(n for n in range(1,45) if n*n<2023 and n*n%5==0),
      (72,1):12,(85,2):None,(216,1):F(-1,2),(216,2):F(-1,2),(216,3):F(-7,8),
      (251,2):2*5**6-2**6,(257,2):sum(b<=41<b*b for b in range(2,42)),
      (268,1):63,(268,2):4022+63,(269,2):math.comb(14,7),
      (273,1):72,(273,2):F(sum(range(1,73)),2),(273,3):F(sum(range(3,73,3)),2),
      (273,4):sum(F(j,2)*(-1 if j%3==0 else 1) for j in range(1,73)),
      (274,1):F(3,5)**3/6,(274,2):F(1,10)**3/3,(274,3):1-8*F(9,250)+12*F(1,3000),
      (285,1):min(p for p in range(1,200) for a in range(1,200) if 500<5*a+5*p<1000 and 1000<2*a+12*p<2000),
      (295,1):next(n for n in range(2024,2100) if sum(map(int,str(n)))==17),
    }
    for (i,j),value in exact.items():
        if value is not None:verify(i,j,value,'Recomputed arithmetic/enumeration')
    values=[n for n in range(1000,10000) if sum(int(d)**2 for d in str(n))==17]
    verify(262,1,(len(values),values[len(values)//2],values[len(values)//2]),'Exhaustive four-digit enumeration')
    verify(262,2,values[len(values)//2],'Median of enumerated set')
    multiple=max(n for n in range(10000,100000) if n%63==0 and len(set(str(n)))==5)
    verify(263,1,multiple,'All five-digit integers tested')
    target=math.prod(range(2,2013,2));product=1;odd=[]
    for n in range(1,2013,2):
        product*=n
        if target%product==0:odd.append(n)
    assert odd==list(range(1,1008,2))
    verify(260,1,2012//2,'Every even prefix divides target; larger ones exceed it')
    verify(260,2,len(odd),'Exact big-integer divisibility; failure from prime1009 onward')
    verify(260,3,1006+len(odd),'Combine disjoint parity classes')
    # Potential argument checked for every configuration up to n=10 (not a substitute for proof).
    for n in range(1,11):
        potentials=[]
        for bits in itertools.product((0,1),repeat=n):
            k=sum(bits);H=sum(i*b for i,b in enumerate(bits,1));P=2*H-k*k
            assert P>=k
            if k:
                nxt=list(bits);nxt[k-1]^=1
                kk=sum(nxt);pp=2*sum(i*b for i,b in enumerate(nxt,1))-kk*kk
                assert pp==P-1
            potentials.append(P)
        assert F(sum(potentials),2**n)==F(n*(n+1),4)
    checks.append(dict(question_number=242,check='Exhaustive potential transitions and expectation for n=1..10; general algebra in reference'))
    found=[]
    def digits(prefix):
        found.append(int(''.join(map(str,prefix))))
        lower=sum(prefix[-2:]) if len(prefix)>=2 else 0
        for d in range(lower,10):digits(prefix+[d])
    for first in range(1,10):digits([first])
    verify(253,1,max(len(str(n)) for n in found),'Enumerated every admissible prefix until no digit extension possible')
    verify(253,2,max(found),'Largest enumerated fibbish integer')
    for n in range(1,13):
        count=0
        for mask in range(1,1<<n):
            seq=[i+1 for i in range(n) if mask>>i&1]
            count+=all(v%2==j%2 for j,v in enumerate(seq,1))
        a,b=0,1
        for _ in range(n+2):a,b=b,a+b
        assert count==a-1
    checks.append(dict(question_number=283,check='Every subset through N=12 agrees with e(N)=F_(N+2)-1'))
    return checks
