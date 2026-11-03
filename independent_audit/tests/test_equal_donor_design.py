from fractions import Fraction
from independent_audit.equal_length_donors import PAIRS


def test_predefined_donor_gold_answers():
    expected=[8+6,24-9,7*9,26+53,24*16,Fraction(16,100)*450]
    assert [Fraction(p[3]) for p in PAIRS]==expected
