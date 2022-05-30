import unittest

from src.fraction import Fraction


class Operation(unittest.TestCase):
    def test1(self):
        fract1 = Fraction(1, 2)
        fract2 = Fraction(3, 6)
        fract3 = Fraction(2, -8)

        assert fract1._numerator == 1
        assert fract1._denominator == 2
        assert fract2._numerator == 1
        assert fract2._denominator == 2
        assert fract3._numerator == -1
        assert fract3._denominator == 4

    def test2(self):
        fract1 = Fraction(1, 2)
        fract2 = Fraction(-5, -10)

        assert fract1 == fract2

    def test3(self):
        fract1 = Fraction(3, 6)
        fract2 = Fraction(1, 4)

        assert fract1 + fract2 == Fraction(3, 4)
