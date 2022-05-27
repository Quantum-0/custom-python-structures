from __future__ import annotations

from collections.abc import Callable
from typing import Tuple, Union, List, Generic, TypeVar, Optional

# Matrix Value Type
_MVT = TypeVar('_MVT')  # Union[int, float, complex]


class NotSquareMatrix(ValueError):
    pass


class Matrix(Generic[_MVT]):
    def __init__(self, width: int, height: int, values: List[List[_MVT]], *, check_sizes: bool = False):
        self._width: int = width
        self._height: int = height
        self._values: List[List[_MVT]] = values
        if check_sizes:
            raise NotImplementedError

    def __getitem__(self, index: Union[Tuple[int, int], Tuple[int, slice], Tuple[slice, int], Tuple[slice, slice]]):
        pass

    def __setitem__(self, key, value: _MVT):
        pass

    def transpose(self) -> Matrix[_MVT]:
        pass

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def size(self) -> Tuple[int, int]:
        return self._width, self._height

    @property
    def is_square(self):
        return self._width == self._height

    @classmethod
    def generate(cls, width, height, value: Union[Callable[[int, int], _MVT], Callable[[], _MVT], _MVT]) -> Matrix[_MVT]:
        """ Generates matrix from size and generator, for example (2, 2, lambda x,y: int(input()) """

    @classmethod
    def from_nested_list(cls, values: List[List[_MVT]]):
        h = len(values)
        if h == 0:
            raise ValueError('Cannot create matrix from empty list')
        w = values[0]
        if not all(len(row) == w for row in values):
            raise ValueError('All rows must have equal length')
        raise NotImplementedError()

    @classmethod
    def from_lists(cls, *lists: List[_MVT]):
        return cls.from_nested_list(values=list(lists))

    @classmethod
    def zero_matrix(cls, size):
        return cls.generate(width=size, height=size, value=0)

    @classmethod
    def identity(cls, size):
        return cls.generate(width=size, height=size, value=lambda x, y: 1 if x == y else 0)

    @classmethod
    def input_matrix(
            cls,
            height: Optional[int] = None,
            width: Optional[int] = None,
            postprocess: Callable[[str]:_MVT] = lambda x: int(x),
            *,
            width_first: bool = False
    ) -> Matrix[_MVT]:
        if width_first:
            height = height or int(input())
            width = width or int(input())
        else:
            width = width or int(input())
            height = height or int(input())
        assert isinstance(width, int)
        assert isinstance(height, int)
        values = list()
        for r in range(height):
            row = list()
            for v in range(width):
                row.append(postprocess(v))
            values.append(row)
        return Matrix(width=width, height=height, values=values)

    @property
    def main_diagonal(self) -> list:
        """ Returns list of main diagonal elements """
        if not self.is_square:
            raise

    @property
    def trace(self):
        return sum(self.main_diagonal)

    def __invert__(self):
        """ Overrides operator ~A """
        raise NotImplementedError()

    def __add__(self, other: Matrix) -> Matrix[_MVT]:
        pass

    def __mul__(self, other: Matrix) -> Matrix[_MVT]:
        pass

    def __eq__(self, other) -> bool:
        pass
