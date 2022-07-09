from __future__ import annotations
from math import gcd
from typing import Union


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

    @classmethod
    def from_float(cls, float: float):
        denominator = 10**(len(str(float)) - len(str(float // 1))+1)
        numerator = int(str(float).replace('.', ''))
        return cls(numerator=numerator, denominator=denominator)

    def __eq__(self, other: Union[Fraction, int, float]) -> bool:

        if isinstance(other, Fraction):
            return self._numerator == other._numerator and self._denominator == other._denominator

        if isinstance(other, (int, float)):
            return self._numerator / self._denominator == other

        raise TypeError

    def __neg__(self):
        return Fraction(-self._numerator, self._denominator)

    def __lt__(self, other: Union[Fraction, int, float]) -> bool:

        if isinstance(other, Fraction):
            return self._numerator * other._denominator < other._numerator * self._denominator

        if isinstance(other, (int, float)):
            return self._numerator / self._denominator < other

    def __le__(self, other: Union[Fraction, int, float]) -> bool:
        return self.__eq__(other) or self.__lt__(other)

    def __gt__(self, other: Union[Fraction, int, float]) -> bool:
        return not self.__le__(other)

    def __ge__(self, other: Union[Fraction, int, float]) -> bool:
        return not self.__lt__(other)

    def __add__(self, other: Union[Fraction, int]) -> Fraction:

        if isinstance(other, Fraction):
            return Fraction(
                self._numerator * other._denominator + other._numerator * self._denominator,
                self._denominator * other._denominator
            )

        if isinstance(other, int):
            return Fraction(
                self._numerator + other * self._denominator,
                self._denominator
            )

        raise TypeError

    def __radd__(self, other: Union[Fraction, int]) -> Fraction:
        return self.__add__(other)

    def __sub__(self, other: Union[Fraction, int]) -> Fraction:

        if isinstance(other, Fraction):
            return Fraction(
                self._numerator * other._denominator - other._numerator * self._denominator,
                self._denominator * other._denominator
            )

        if isinstance(other, int):
            return Fraction(
                self._numerator - other * self._denominator,
                self._denominator
            )

        raise TypeError

    def __rsub__(self, other: Union[Fraction, int]) -> Fraction:
        return -self.__sub__(other)

    def __mul__(self, other: Union[Fraction, int]) -> Fraction:

        if isinstance(other, Fraction):
            return Fraction(self._numerator * other._numerator, self._denominator * other._denominator)

        if isinstance(other, int):
            return Fraction(self._numerator * other, self._denominator)

        raise TypeError

    def __rmul__(self, other: Union[Fraction, int]) -> Fraction:
        return self.__mul__(other)

    def __truediv__(self, other: Union[Fraction, int]) -> Fraction:

        if isinstance(other, Fraction):
            return Fraction(self._numerator * other._denominator, self._denominator * other._numerator)

        if isinstance(other, int):
            return Fraction(self._numerator, self._denominator * other)

        raise TypeError

    def __rtruediv__(self, other: Union[Fraction, int]) -> Fraction:

        if isinstance(other, Fraction):
            return Fraction(self._denominator * other._numerator, self._numerator * other._denominator)

        if isinstance(other, int):
            return Fraction(self._denominator * other, self._numerator)

        raise TypeError

    def __repr__(self):
        return f"{self._numerator}/{self._denominator}"
