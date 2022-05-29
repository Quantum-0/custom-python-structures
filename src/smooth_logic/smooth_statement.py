from __future__ import annotations
from typing import Optional, Union


class InvalidChanceValue(ValueError):
    pass


# TODO: Dependency chances (tree probably, dunno)
# TODO: fraction (for a more accurate comparison)
# TODO: Unknown Chance
# TODO: Rename to "Chance"
class SmoothStatement:
    def __init__(self, chance: Union[float, int, bool, str], *, weight: float = 1):
        if isinstance(chance, (float, int)):
            if not 0 <= chance <= 1:
                raise InvalidChanceValue("Chance must be between 0 and 1")
            self._chance = chance
        elif isinstance(chance, bool):
            self._chance = 1 if chance else 0
        elif isinstance(chance, str):
            if chance[-1] != '%':
                raise InvalidChanceValue("Chance format is X%")
            self._chance = float(chance[:-1]) / 100
            if not 0 < self._chance < 1:
                raise InvalidChanceValue("Chance must be between 0% and 100%")
        else:
            raise InvalidChanceValue
        self._weight = weight

    @property
    def chance(self) -> float:
        return self._chance

    def __float__(self) -> float:
        return self.chance

    @property
    def percent(self) -> float:
        return 100 * self._chance

    def __bool__(self) -> Optional[bool]:
        return True if self._chance == 1 else False if self._chance == 0 else None

    def __and__(self, other: object) -> SmoothStatement:
        if isinstance(other, bool):
            return SmoothStatement(self._chance if other else 0, weight=self._weight + 1)
        elif isinstance(other, SmoothStatement):
            if other is self:
                return self
            new_chance = round(self._chance * other.chance, 10)
            return SmoothStatement(new_chance, weight=self._weight + other._weight)  # TODO: weights
        else:
            raise AttributeError

    def __or__(self, other: object) -> SmoothStatement:
        if isinstance(other, bool):
            return SmoothStatement(1 if other else self._chance, weight=self._weight + 1)
        elif isinstance(other, SmoothStatement):
            if other is self:
                return self
            new_chance = round(1 - (1 - self._chance) * (1 - other.chance), 10)
            return SmoothStatement(new_chance, weight=self._weight + other._weight)  # TODO: weights
        else:
            raise AttributeError

    def __invert__(self) -> SmoothStatement:
        return SmoothStatement(1-self._chance, weight=self._weight)

    def __xor__(self, other: object) -> SmoothStatement:
        if isinstance(other, bool):
            return SmoothStatement(1-self._chance if other else self._chance, weight=self._weight + 1)
        elif isinstance(other, SmoothStatement):
            if other is self:
                return SmoothStatement(0, weight=self._weight*2)
            new_chance = round(self._chance + other._chance - 2 * self._chance * other.chance, 10)
            return SmoothStatement(new_chance, weight=self._weight + other._weight)  # TODO: weights
        else:
            raise AttributeError

    def __lt__(self, other) -> bool:
        if isinstance(other, bool):
            return self._chance < int(other)
        elif isinstance(other, SmoothStatement):
            return self._chance < other.chance
        elif isinstance(other, float):
            return self._chance < other
        else:
            raise AttributeError

    def __le__(self, other) -> bool:
        if isinstance(other, bool):
            return self._chance <= int(other)
        elif isinstance(other, SmoothStatement):
            return self._chance <= other.chance
        elif isinstance(other, float):
            return self._chance <= other
        else:
            raise AttributeError

    def __gt__(self, other) -> bool:
        return not self.__lt__(other)

    def __ge__(self, other) -> bool:
        return not self.__le__(other)

    def __eq__(self, other):
        if isinstance(other, bool):
            return self._chance == int(other)
        elif isinstance(other, SmoothStatement):
            return round(abs(self._chance - other.chance), 10) == 0
        elif isinstance(other, float):
            return round(abs(self._chance - other), 10) == 0
        else:
            raise AttributeError

    def __mul__(self, other) -> SmoothStatement:
        if isinstance(other, int):
            return SmoothStatement(chance=self._chance ** other, weight=self._weight * other)

    def __repr__(self):
        # TODO: print as fraction
        return f"<{self.__class__.__name__}({str(self._chance)+'%' if 0.02 <= self._chance <= 0.98 else 'TODO'})>"


class Constants:
    FALSE = SmoothStatement(0)
    TRUE = SmoothStatement(1)
    NO = SmoothStatement(0)
    YES = SmoothStatement(1)
    NEVER = SmoothStatement(0)
    ALWAYS = SmoothStatement(1)
    MAYBE = SmoothStatement(0.5)
    SOMETIMES = SmoothStatement(0.5)
    HALF = SmoothStatement(0.5)
    UNLIKELY = SmoothStatement(0.4)
    LIKELY = SmoothStatement(0.6)
    RARELY = SmoothStatement(0.2)
    OFTEN = SmoothStatement(0.8)
