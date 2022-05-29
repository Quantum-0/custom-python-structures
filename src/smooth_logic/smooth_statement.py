from __future__ import annotations
from typing import Optional, Union


class InvalidChanceValue(ValueError):
    pass


class _UnknownChance:
    def __bool__(self):
        return False

    def __copy__(self):
        return self

    def __deepcopy__(self, _):
        return self

    def __repr__(self):
        return f"<{Chance.__name__}(Unknown)>"

    @property
    def chance(self) -> None:
        return None

    @property
    def percent(self) -> None:
        return None


# Singleton value indicates that value of chance is unknown
unknown_chance = _UnknownChance()


# TODO: Dependency chances (tree probably, dunno)
# TODO: fraction (for a more accurate comparison)
# TODO: Unknown Chance
class Chance:
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

    def __and__(self, other: object) -> Chance:
        if isinstance(other, bool):
            return Chance(self._chance if other else 0, weight=self._weight + 1)
        elif isinstance(other, Chance):
            if other is self:
                return self
            new_chance = round(self._chance * other.chance, 10)
            return Chance(new_chance, weight=self._weight + other._weight)  # TODO: weights
        else:
            raise AttributeError

    def __or__(self, other: object) -> Chance:
        if isinstance(other, bool):
            return Chance(1 if other else self._chance, weight=self._weight + 1)
        elif isinstance(other, Chance):
            if other is self:
                return self
            new_chance = round(1 - (1 - self._chance) * (1 - other.chance), 10)
            return Chance(new_chance, weight=self._weight + other._weight)  # TODO: weights
        else:
            raise AttributeError

    def __invert__(self) -> Chance:
        return Chance(1 - self._chance, weight=self._weight)

    def __xor__(self, other: object) -> Chance:
        if isinstance(other, bool):
            return Chance(1 - self._chance if other else self._chance, weight=self._weight + 1)
        elif isinstance(other, Chance):
            if other is self:
                return Chance(0, weight=self._weight * 2)
            new_chance = round(self._chance + other._chance - 2 * self._chance * other.chance, 10)
            return Chance(new_chance, weight=self._weight + other._weight)  # TODO: weights
        else:
            raise AttributeError

    def __lt__(self, other) -> bool:
        if isinstance(other, bool):
            return self._chance < int(other)
        elif isinstance(other, Chance):
            return self._chance < other.chance
        elif isinstance(other, float):
            return self._chance < other
        else:
            raise AttributeError

    def __le__(self, other) -> bool:
        if isinstance(other, bool):
            return self._chance <= int(other)
        elif isinstance(other, Chance):
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
        elif isinstance(other, Chance):
            return round(abs(self._chance - other.chance), 10) == 0
        elif isinstance(other, float):
            return round(abs(self._chance - other), 10) == 0
        else:
            raise AttributeError

    def __mul__(self, other) -> Chance:
        if isinstance(other, int):
            return Chance(chance=self._chance ** other, weight=self._weight * other)

    def __repr__(self):
        # TODO: print as fraction
        return f"<{self.__class__.__name__}({str(self._chance)+'%' if 0.02 <= self._chance <= 0.98 else 'TODO'})>"


class Constants:
    FALSE = Chance(0)
    TRUE = Chance(1)
    NO = Chance(0)
    YES = Chance(1)
    NEVER = Chance(0)
    ALWAYS = Chance(1)
    MAYBE = Chance(0.5)
    SOMETIMES = Chance(0.5)
    HALF = Chance(0.5)
    UNLIKELY = Chance(0.4)
    LIKELY = Chance(0.6)
    RARELY = Chance(0.2)
    OFTEN = Chance(0.8)
