from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Tuple, Union, List, Generic, TypeVar, Optional, Any

# Matrix Value Type
_MVT = TypeVar('_MVT')  # Union[int, float, complex] # + maybe bool


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
        # Type Check
        if not isinstance(index, tuple):
            raise IndexError("Index must be tuple")
        if len(index) != 2:
            raise IndexError("Index must be tuple of two values")
        if not isinstance(index[0], (int, slice)) or not isinstance(index[1], (int, slice)):
            raise IndexError("Index must be tuple of int/slice values")
        # Bounds
        if (isinstance(index[0], int) and (index[0] < 0 or index[0] >= self._width)) or \
            (isinstance(index[1], int) and (index[1] < 0 or index[1] >= self._height)):
            #(isinstance(index[0], slice) and (index[0] and)):
            raise IndexError("Matrix doesn't support negative or overflow indexes")
        # if (isinstance(index[0], slice) and index[0].step != None) or (isinstance(index[1], slice) and index[1].step != None):
        #     raise IndexError("Matrix doesn't support slice step")

        return self._values[index[1]][index[0]]


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
    def generate(
            cls,
            width,
            height,
            value: Union[Callable[[int, int], _MVT], Callable[[], _MVT], _MVT, Iterator]
    ) -> Matrix[_MVT]:
        """ Generates matrix from size and generator, for example (2, 2, lambda x,y: x+y """
        values = list()
        for y in range(height):
            row = list()
            for x in range(width):
                if callable(value):
                    if value.__code__.co_argcount == 2:
                        row.append(value(x, y))
                    elif value.__code__.co_argcount == 0:
                        row.append(value())
                    else:
                        raise ValueError('Incorrect number of arguments for generator')
                elif isinstance(value, Iterator):
                    row.append(next(value))
                else:
                    row.append(value)
            values.append(row)
        return Matrix(width=width, height=height, values=values)

    @classmethod
    def from_nested_list(cls, values: List[List[_MVT]]):
        h = len(values)
        if h == 0:
            raise ValueError('Cannot create matrix from empty list')
        w = len(values[0])
        if not all(len(row) == w for row in values):
            raise ValueError('All rows must have equal length')
        return Matrix(w, h, values, check_sizes=False)

    @classmethod
    def from_joined_lists(cls, width: int, height: int = None, *, values: List[_MVT] or range) -> Matrix[_MVT]:
        lists = list()
        if isinstance(values, range):
            values = list(values)
        for i in range(0, len(values), width):
            lists.append(values[i:i+width])
        if height and len(lists) != height or len(lists[-1]) != width:
            raise ValueError('Incorrect elements count')
        return cls.from_nested_list(lists)

    @classmethod
    def from_lists(cls, *lists: List[_MVT]) -> Matrix[_MVT]:
        return cls.from_nested_list(values=list(lists))

    @classmethod
    def zero_matrix(cls, size) -> Matrix[_MVT]:
        return cls.generate(width=size, height=size, value=0)

    @classmethod
    def identity(cls, size) -> Matrix[_MVT]:
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
        return cls.generate(width, height, lambda: postprocess(input()))

    @property
    def main_diagonal(self) -> list:
        """ Returns list of main diagonal elements """
        if not self.is_square:
            raise

    @property
    def trace(self) -> Union[int, Any]:
        return sum(self.main_diagonal)

    def __contains__(self, item: Matrix[_MVT] or List[List[_MVT]]) -> bool:
        other = Matrix.from_nested_list(item) if isinstance(item, List) else item
        if other.width > self.width or other.height > self.height:
            return False
        offsets_x = range(self.width - other.width + 1)
        offsets_y = range(self.height - other.height + 1)
        for offset_y in offsets_y:
            for offsets_x in offsets_x:
                eq = True
                for y in range(other.height):
                    for x in range(other.width):
                        if self._values[y][x] != other._values[y][x]:
                            eq = False
                            break
                    if not eq:
                        break
                if eq:
                    return True
        return False

    def __invert__(self):
        """ Overrides operator ~A """
        raise NotImplementedError()

    def __add__(self, other: Matrix) -> Matrix[_MVT]:
        pass

    def __mul__(self, other: Matrix) -> Matrix[_MVT]:
        pass

    def __eq__(self, other: List[List[_MVT]] or Matrix[_MVT]) -> bool:
        if isinstance(other, Matrix):
            return self.size == other.size and self._values == other._values
        else:
            try:
                return self._values == other
            except ValueError:
                return False
