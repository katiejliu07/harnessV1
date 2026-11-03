"""Fresh, fixed development/validation tasks; do not tune on validation outputs.

Expected values are evaluator-only. Runtime infer() receives just prompt/kind,
never these answers/cases. Exact Fraction arithmetic avoids float-derived gold
errors in percentages, taxes and fractions. Original 37 tasks are regressions,
not fresh validation, and are kept outside these selection sets.
"""
from fractions import Fraction
import random


def arithmetic(seed,rounds):
    rng=random.Random(seed)
    tasks=[]
    for turn in range(rounds):
        a,b=rng.randint(112,389),rng.randint(41,198)
        m,n=rng.randint(16,43),rng.randint(11,29)
        num,den=rng.choice([(5,8),(7,12),(11,16),(9,20)])
        whole=rng.randint(11,29)*den
        rate,amount=rng.randint(11,24),rng.randint(12,48)*10
        coeff,x,offset=rng.randint(3,9),rng.randint(8,27),rng.randint(5,19)
        p,discount,tax=rng.randint(8,27)*10,rng.choice([10,15,20,25]),rng.choice([5,6,8])
        h1,h2=rng.randint(2,5),rng.randint(2,5)
        d1,d2=rng.randint(31,83)*h1,rng.randint(24,67)*h2
        s,t=rng.randint(32,59),rng.randint(11,27)
        j,k,l=rng.randint(5,18),rng.randint(4,15),rng.randint(3,12)
        first,second=rng.randint(2,5),rng.randint(3,7)
        total=(first+second)*rng.randint(9,28)
        u,v=rng.randint(8,29),rng.randint(7,21)
        tenths=rng.randint(13,68)
        cases=[
            (f"What is {a} plus {b}?",a+b),
            (f"What is {m} times {n}?",m*n),
            (f"What is {num}/{den} of {whole}?",Fraction(num,den)*whole),
            (f"What is {rate} percent of {amount}?",Fraction(rate,100)*amount),
            (f"Solve {coeff}x + {offset} = {coeff*x+offset} for x.",x),
            (f"An item costs {p} dollars before a {discount} percent discount and then {tax} percent tax. What is its exact final price in dollars?",Fraction(p)*(1-Fraction(discount,100))*(1+Fraction(tax,100))),
            (f"A vehicle travels {d1} miles in {h1} hours, then {d2} miles in {h2} hours. What is the overall average speed in miles per hour? Give an exact number or fraction.",Fraction(d1+d2,h1+h2)),
            (f"What is {s} squared minus {t} squared?",s*s-t*t),
            (f"Evaluate {j} + {k} * {l}, using standard order of operations.",j+k*l),
            (f"A total of {total} is split in the ratio {first}:{second}. What is the first share?",Fraction(total*first,first+second)),
            (f"What is (-{u}) times (-{v})?",u*v),
            (f"How many minutes are in {tenths//10}.{tenths%10} hours?",Fraction(tenths,10)*60),
        ]
        tasks.extend({"id":f"arith_{turn:02d}_{i:02d}","kind":"scalar","prompt":q,"answer":str(Fraction(y))} for i,(q,y) in enumerate(cases))
    return tasks


def code(name,prompt,cases):
    return {"id":"code_"+name,"kind":"code","function":name,"prompt":prompt,"cases":cases}


DEV_CODE=[
    code("interleave","Write interleave(a, b), a Python function returning a new list that alternates items from a and b, starting with a, and appends the remaining items of a longer input. Handle empty inputs. Do not modify inputs.",
         [([[1,2],[8,9]],[1,8,2,9]),([[1,2,3],[4]],[1,4,2,3]),([[],[1,2]],[1,2]),([[],[]],[])]),
    code("clip_values","Write clip_values(values, lo, hi), returning a new list with each numeric value clipped to the inclusive range [lo, hi]. Assume lo <= hi. Preserve order, handle empty input and do not modify inputs.",
         [([[-3,0,4,9],0,5],[0,0,4,5]),([[],1,2],[]),([[2,3],2,2],[2,2]),([[-4,-1],-3,-2],[-3,-2])]),
    code("adjacent_differences","Write adjacent_differences(values), returning a new list of each value minus its predecessor, in order. Lists with fewer than two values return []. Do not modify the input.",
         [([[4,7,2,2]],[3,-5,0]),([[]],[]),([[8]],[]),([[-3,-7,2]],[-4,9])]),
    code("even_first","Write even_first(values), returning a new list with all even integers followed by all odd integers, preserving original order within each group. Handle empty input. Do not modify the input.",
         [([[3,2,1,4,2]],[2,4,2,3,1]),([[]],[]),([[-3,-2,0,5]],[-2,0,-3,5]),([[1,3]],[1,3])]),
]
VALIDATION_CODE=[
    code("rotate_right","Write rotate_right(values, k), returning a new list rotated right by integer k. Negative k rotates left. Handle empty input and k larger than the length. Do not modify the input.",
         [([[1,2,3,4],1],[4,1,2,3]),([[1,2,3],5],[2,3,1]),([[1,2,3],-1],[2,3,1]),([[],4],[]),([[1,2],0],[1,2])]),
    code("merge_sorted","Write merge_sorted(a, b), returning a new sorted list containing all values from the two already sorted lists, including duplicates. Handle empty inputs and do not modify inputs.",
         [([[1,3,3],[2,3,4]],[1,2,3,3,3,4]),([[],[]],[]),([[],[1,2]],[1,2]),([[-5,0],[-3,2]],[-5,-3,0,2])]),
    code("prefix_sums","Write prefix_sums(values), returning a new list of inclusive running sums. Handle empty input and negative numbers. Do not modify the input.",
         [([[2,3,-1]],[2,5,4]),([[]],[]),([[-2,2,4]],[-2,0,4]),([[0,0]],[0,0])]),
    code("moving_sums","Write moving_sums(values, k), returning the sums of all consecutive windows of exactly k elements. Return [] for k <= 0 or k > len(values). Handle empty input. Do not modify inputs.",
         [([[1,2,3,4],2],[3,5,7]),([[2,-1,5],1],[2,-1,5]),([[1,2],3],[]),([[],1],[]),([[1,2],0],[]),([[1,2],-2],[])]),
    code("expand_runs","Write expand_runs(pairs), returning a new list that expands each (value, count) pair into count copies of value in order. Counts are nonnegative integers. Handle zero counts and empty input. Do not modify inputs.",
         [([[['a',2],['b',0],['c',1]]],['a','a','c']),([[]],[]),([[[3,1],[3,2]]],[3,3,3]),([[[0,0]]],[])]),
    code("nonzero_indices","Write nonzero_indices(values), returning a list of indices of all nonzero numbers, in original order. Handle empty input. Do not modify the input.",
         [([[0,3,0,-2]],[1,3]),([[]],[]),([[0,0]],[]),([[2,-1]],[0,1])]),
    code("flatten_once","Write flatten_once(rows), returning a new list made by concatenating the lists in rows in order, flattening exactly one level. Handle empty rows and empty input. Do not modify inputs.",
         [([[[1,2],[],[3]]],[1,2,3]),([[]],[]),([[[],[]]],[]),([[[[1]],[2]]],[[1],2])]),
    code("keep_last","Write keep_last(items), returning a list containing only the last occurrence of each hashable item, in the order of those last occurrences. Handle empty input. Do not modify the input.",
         [([[1,2,1,3,2]],[1,3,2]),([[]],[]),([['a','b','a']],['b','a']),([[None,False,None]],[False,None])]),
]
DEV=arithmetic(202610291,1)+DEV_CODE
VALIDATION=arithmetic(202610292,2)+VALIDATION_CODE
