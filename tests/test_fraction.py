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
        fract3 = Fraction(4, 2)

        assert fract1 == fract2
        assert fract3 == 2
        assert fract1 == 0.5

    def test3(self):
        fract1 = Fraction(3, 6)
        fract2 = Fraction(1, 4)

        assert fract1 + fract2 == Fraction(3, 4)
        assert fract1 + 5 == Fraction(11, 2)
        assert 5 + fract1 == Fraction(11, 2)
        assert 8 + fract1 == fract1 + 3 + 5
        assert fract1 + fract2 == fract2 + fract1


    def test4(self):
        fract1 = Fraction(3, 6)
        fract2 = Fraction(1, 4)

        assert fract1 - fract2 == Fraction(1, 4)
        assert fract1 - 1 == Fraction(-1, 2)
        assert 5 - fract1 == 4.5
        assert -fract1 == Fraction(-1, 2)

    def test5(self):
        fract1 = Fraction(3, 6)
        fract2 = Fraction(1, 4)

        assert fract1 * fract2 == Fraction(1, 8)
        assert fract1 * 3 == Fraction(3, 2)
        assert 6 * fract2 == Fraction(3, 2)
        assert fract1 * 8 == 8 * fract1
        assert fract1 * 10 == 5

    def test6(self):
        fract1 = Fraction(1, 4)
        fract2 = Fraction(1, 2)

        assert fract1 / fract2 == Fraction(1, 2)
        assert fract2 / fract1 == 2
        assert fract1 / 2 == Fraction(1, 8)
        assert 2 / fract1 == 8
