import unittest

from src.fraction import Fraction


class Operation(unittest.TestCase):
    def setUp(self) -> None:
        self.fract1_2 = Fraction(1, 2)
        self.fract3_6 = Fraction(3, 6)
        self.fract2_n8 = Fraction(2, -8)
        self.fractn5_n10 = Fraction(-5, -10)
        self.fract4_2 = Fraction(4, 2)
        self.fract1_4 = Fraction(1, 4)
        self.fractn1_2 = Fraction(-1, 2)

    def test_init(self):
        assert self.fract1_2._numerator == 1
        assert self.fract1_2._denominator == 2
        assert self.fract3_6._numerator == 1
        assert self.fract3_6._denominator == 2
        assert self.fract2_n8._numerator == -1
        assert self.fract2_n8._denominator == 4

    def test_eq(self):
        assert self.fract1_2 == self.fractn5_n10
        assert self.fract4_2 == 2
        assert self.fract1_2 == 0.5
        assert 2 == self.fract4_2
        assert self.fract1_2 != 5

    def test_cmp(self):
        assert self.fract1_2 < self.fract4_2
        assert self.fract1_2 < 5
        assert self.fract1_2 < 6.7
        assert self.fract2_n8 < self.fract1_4
        assert self.fractn1_2 < self.fract2_n8
        assert self.fract4_2 <= 2
        assert self.fractn1_2 <= self.fract2_n8
        assert 5 > self.fract1_4
        assert 2 >= self.fract4_2
        assert self.fract1_4 > self.fract2_n8
        assert self.fract2_n8 >= self.fractn1_2
        assert self.fract1_4 >= 0.25
        assert self.fract4_2 >= 1.5

    def test_add(self):
        assert self.fract3_6 + self.fract1_4 == Fraction(3, 4)
        assert self.fract3_6 + 5 == Fraction(11, 2)
        assert 5 + self.fract3_6 == Fraction(11, 2)
        assert 8 + self.fract3_6 == self.fract3_6 + 3 + 5
        assert self.fract3_6 + self.fract1_4 == self.fract1_4 + self.fract3_6

    def test_sub(self):
        assert self.fract3_6 - self.fract1_4 == Fraction(1, 4)
        assert self.fract3_6 - 1 == Fraction(-1, 2)
        assert 5 - self.fract3_6 == 4.5
        assert -self.fract3_6 == Fraction(-1, 2)

    def test_mul(self):
        assert self.fract3_6 * self.fract1_4 == Fraction(1, 8)
        assert self.fract3_6 * 3 == Fraction(3, 2)
        assert 6 * self.fract1_4 == Fraction(3, 2)
        assert self.fract3_6 * 8 == 8 * self.fract3_6
        assert self.fract3_6 * 10 == 5

    def test_div(self):
        assert self.fract1_4 / self.fract1_2 == Fraction(1, 2)
        assert self.fract1_2 / self.fract1_4 == 2
        assert self.fract1_4 / 2 == Fraction(1, 8)
        assert 2 / self.fract1_4 == 8

    def test_from_float(self):
        assert Fraction(1, 4) == Fraction.from_float(1 / 4)
        assert Fraction(5, 2) == Fraction.from_float(5 / 2)
        assert Fraction(100, 4) == Fraction.from_float(100 / 4)
        assert Fraction(16432, 10000) == Fraction.from_float(1.6432)
        assert Fraction(1007933058582, 10000000000) == Fraction.from_float(100.7933058582)
