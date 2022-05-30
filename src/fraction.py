from __future__ import annotations
from math import gcd


class Fraction:

    def __init__(self, numerator: int, denominator: int):
        self._numerator: int = numerator
        self._denominator: int = denominator
        if self._denominator == 0:
            raise ZeroDivisionError
        self.reduction()

    def reduction(self):
        frac_gcd = gcd(self._numerator, self._denominator)
        self._numerator = self._numerator // frac_gcd
        self._denominator = self._denominator // frac_gcd
        if self._denominator < 0:
            self._numerator = -self._numerator
            self._denominator = -self._denominator

    def __eq__(self, other: Fraction):
        return self._numerator == other._numerator and self._denominator == other._denominator

    def __add__(self, other: Fraction):
        res_num = self._numerator * other._denominator + other._numerator * self._denominator
        res_den = self._denominator * other._denominator
        res = Fraction(res_num, res_den)
        return res

    def __sub__(self, other: Fraction):
        res_num = self._numerator * other._denominator - other._numerator * self._denominator
        res_den = self._denominator * other._denominator
        res = Fraction(res_num, res_den)
        return res

    def __mul__(self, other: Fraction):
        res_num = self._numerator * other._numerator
        res_den = self._denominator * other._denominator
        res = Fraction(res_num, res_den)
        return res

    def __truediv__(self, other: Fraction):
        res_num = self._numerator * other._denominator
        res_den = self._denominator * other._numerator
        res = Fraction(res_num, res_den)
        return res

    def __repr__(self):
        return f"{self._numerator}/{self._denominator}"
