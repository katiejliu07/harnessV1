"""Supplemental scalar scorer; frozen benchmark scores are never overwritten.

The first independent scorer could treat 'our answer is correct' as an answer
anchor and stop before a later conclusion. Only explicit Final-answer anchors
take precedence here; otherwise inspect the last numerical paragraph. Keep this
regression because generic 'answer is' matching has now failed twice in the audit.
Scalar scoring still does not assess the validity of an explanation.
"""
import re
from .evaluation import canonical

NUMBER=r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?:/\d+)?"


def extract_answer(text):
    clean=text.replace("−","-").replace("**","").replace("$","")
    clean=re.sub(r"\\frac\{([-+]?\d+)\}\{(\d+)\}",r"\1/\2",clean)
    clean=re.sub(r"\\(?:boxed|text|mathrm)\{([^{}]*)\}",r"\1",clean)
    anchors=list(re.finditer(r"\bfinal\s+answer(?:\s+is)?\s*[:=]?\s*",clean,re.I))
    if anchors:
        clause=clean[anchors[-1].end():].strip()
        clause=re.split(r"\n\s*\n",clause,maxsplit=1)[0]
        clause=re.split(r"(?<=[.!?])\s+(?=[A-Z])",clause,maxsplit=1)[0]
    else:
        paragraphs=[p for p in re.split(r"\n\s*\n",clean) if re.search(NUMBER,p)]
        clause=paragraphs[-1] if paragraphs else ""
    values=re.findall(NUMBER,clause)
    return canonical(values[-1]) if values else None


def correct(text,expected):
    return extract_answer(text)==canonical(expected)
