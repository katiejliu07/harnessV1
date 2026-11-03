"""Hard held-out set: multi-step word problems and harder coding (final exam only).

Used once, to compare no harness against the harness; never to choose or tune a
prompt, so it cannot be overfit. Answers are exact Fractions, evaluator-only.
Code tasks return lists, as the audited grader requires, and get EvalPlus-style
differential tests from lean_harness.differential.
"""
from fractions import Fraction
import random

from .tasks import code


def word_problems(seed=202610296,rounds=2):
    rng=random.Random(seed)
    tasks=[]
    for turn in range(rounds):
        stock,pct,more,boxes,per=rng.randint(30,90)*10,rng.choice([10,20,25,40]),rng.randint(15,60),rng.randint(3,9),rng.randint(12,30)
        a,b=rng.randint(2,9),rng.randint(3,12)
        x,p,y,q=rng.randint(2,9),rng.choice([10,20,30,40]),rng.randint(2,9),rng.choice([50,60,70,80])
        principal,rate,years,deposit=rng.randint(10,50)*100,rng.choice([3,4,5,6]),rng.randint(2,6),rng.randint(2,9)*100
        older,gap,ahead=rng.randint(30,50),rng.randint(20,28),rng.randint(5,12)
        d1,s1,stop,d2,s2=rng.randint(4,9)*30,rng.choice([40,45,60]),rng.choice([15,20,30,45]),rng.randint(3,8)*20,rng.choice([30,40,60])
        up,down=rng.choice([10,20,25,50]),rng.choice([10,20,25,40])
        price,disc,coupon,tax=rng.randint(8,30)*10,rng.choice([10,15,20,25]),rng.randint(1,4)*5,rng.choice([5,6,8])
        workers,days,more_w=rng.randint(4,9),rng.randint(6,15),rng.randint(2,6)
        n,first,diff=rng.randint(8,20),rng.randint(2,9),rng.randint(3,7)
        red,blue,green=rng.randint(3,8),rng.randint(2,7),rng.randint(1,6)
        litres,leak,fill=rng.randint(40,90),rng.randint(2,6),rng.randint(8,15)
        cases=[
            (f"A shop has {stock} items. It sells {pct} percent of them on Monday and {more} more on Tuesday, then receives {boxes} boxes of {per} items each. How many items does it have now?",
             stock-Fraction(pct,100)*stock-more+boxes*per),
            (f"Pipe A fills a tank in {a} hours and pipe B fills it in {b} hours. Working together, how many hours do they take? Give an exact number or fraction.",
             Fraction(a*b,a+b)),
            (f"{x} litres of a {p} percent salt solution are mixed with {y} litres of a {q} percent salt solution. What percent salt is the mixture? Give an exact number or fraction.",
             Fraction(x*p+y*q,x+y)),
            (f"You invest {principal} dollars at {rate} percent simple interest per year for {years} years, and add {deposit} dollars at the end. How many dollars do you have in total?",
             principal+Fraction(principal*rate*years,100)+deposit),
            (f"A parent is {older} years old and their child is {older-gap}. In how many years will the parent be exactly twice the child's age? If that happened in the past, give a negative number.",
             Fraction(older-2*(older-gap))),
            (f"A car drives {d1} miles at {s1} mph, stops for {stop} minutes, then drives {d2} miles at {s2} mph. What is its average speed in mph over the whole trip including the stop? Give an exact number or fraction.",
             Fraction(d1+d2)/(Fraction(d1,s1)+Fraction(stop,60)+Fraction(d2,s2))),
            (f"A rectangle's length is increased by {up} percent and its width is decreased by {down} percent. By what percent does its area change? Use a negative number for a decrease and give an exact number or fraction.",
             (Fraction(100+up,100)*Fraction(100-down,100)-1)*100),
            (f"A jacket costs {price} dollars. It is discounted {disc} percent, then a {coupon} dollar coupon is applied, then {tax} percent tax is added. What is the final price in dollars? Give an exact number.",
             (price*Fraction(100-disc,100)-coupon)*Fraction(100+tax,100)),
            (f"{workers} workers finish a job in {days} days. Working at the same rate, how many days would {workers+more_w} workers take? Give an exact number or fraction.",
             Fraction(workers*days,workers+more_w)),
            (f"An arithmetic sequence starts at {first} and increases by {diff} each term. What is the sum of the first {n} terms?",
             Fraction(n*(2*first+(n-1)*diff),2)),
            (f"A bag holds {red} red, {blue} blue and {green} green marbles. Two marbles are drawn without replacement. What is the probability both are red? Give an exact fraction.",
             Fraction(red*(red-1),(red+blue+green)*(red+blue+green-1))),
            (f"A {litres} litre tank is empty. A tap adds {fill} litres per minute while a leak removes {leak} litres per minute. How many minutes until it is full? Give an exact number or fraction.",
             Fraction(litres,fill-leak)),
        ]
        tasks.extend({"id":f"hard_{turn:02d}_{i:02d}","kind":"scalar","prompt":q,"answer":str(Fraction(v))} for i,(q,v) in enumerate(cases))
    return tasks


HARD_CODE=[
    code("merge_intervals","Write merge_intervals(intervals), where intervals is a list of [start, end] pairs with start <= end in any order. Return a new list of merged [start, end] pairs sorted by start; intervals that overlap or touch (one ends where the next starts) merge. Handle empty input. Do not modify the input.",
         [([[[1,3],[2,6],[8,10]]],[[1,6],[8,10]]),([[]],[]),([[[5,7],[1,2],[2,4]]],[[1,4],[5,7]]),([[[1,10],[2,3]]],[[1,10]])]),
    code("spiral_order","Write spiral_order(matrix), returning a list of the elements of a rectangular matrix (a list of equal-length rows) in clockwise spiral order starting at the top-left. Handle an empty matrix and empty rows. Do not modify the input.",
         [([[[1,2,3],[4,5,6],[7,8,9]]],[1,2,3,6,9,8,7,4,5]),([[]],[]),([[[1,2,3,4]]],[1,2,3,4]),([[[1],[2],[3]]],[1,2,3]),([[[1,2],[3,4],[5,6]]],[1,2,4,6,5,3])]),
    code("top_k_frequent","Write top_k_frequent(words, k), returning the k most frequent words, ordered by count descending and then alphabetically for ties. Return all distinct words if there are fewer than k; return [] for k <= 0. Do not modify the input.",
         [([["b","a","b","c","a","b"],2],["b","a"]),([["x","y"],5],["x","y"]),([[],3],[]),([["a","b"],0],[]),([["d","c","d","c"],1],["c"])]),
    code("next_greater","Write next_greater(values), returning a list where each position holds the first value to its right that is strictly greater, or None if there is none. Handle empty input. Do not modify the input.",
         [([[2,1,3,2,4]],[3,3,4,4,None]),([[]],[]),([[5,5,5]],[None,None,None]),([[3,2,1]],[None,None,None])]),
    code("primes_upto","Write primes_upto(n), returning a list of all prime numbers less than or equal to n in increasing order; return [] when there are none.",
         [([10],[2,3,5,7]),([1],[]),([2],[2]),([-5],[]),([30],[2,3,5,7,11,13,17,19,23,29])]),
    code("rle_encode","Write rle_encode(text), returning a list of [character, count] pairs for each run of identical consecutive characters, in order. Return [] for an empty string.",
         [(["aaabcc"],[["a",3],["b",1],["c",2]]),([""],[]),(["x"],[["x",1]]),(["abab"],[["a",1],["b",1],["a",1],["b",1]])]),
    code("sliding_max","Write sliding_max(values, k), returning the maximum of every window of exactly k consecutive values, in order. Return [] for k <= 0 or k > len(values). Do not modify the input.",
         [([[1,3,-1,-3,5,3,6,7],3],[3,3,5,5,6,7]),([[4],1],[4]),([[1,2],3],[]),([[],1],[]),([[1,2],0],[])]),
    code("tokenize_expr","Write tokenize_expr(text), returning a list of tokens from an arithmetic expression: non-negative integers as int values and each of + - * / ( ) as a one-character string, skipping spaces.",
         [(["12+(3*4)"],[12,"+","(",3,"*",4,")"]),([""],[]),([" 7 "],[7]),(["10 / 2-0"],[10,"/",2,"-",0])]),
]
HARD=word_problems()+HARD_CODE
