from __future__ import annotations

import numbers
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

    def __getitem__(self, index: Union[Tuple[int, int], Tuple[int, slice], Tuple[slice, int], Tuple[slice, slice]]) -> _MVT:
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

        # Y is int
        if isinstance(index[1], int):
            return self._values[index[1]][index[0]][:] if isinstance(index[0], slice) else self._values[index[1]][index[0]]
        else:
            return [row[index[0]] for row in self._values[index[1]]]


    def __setitem__(
            self,
            key: Union[Tuple[int, int], Tuple[int, slice], Tuple[slice, int], Tuple[slice, slice]],
            value: _MVT):
        # Type Check
        if not isinstance(key, tuple):
            raise IndexError("Index must be tuple")
        if len(key) != 2:
            raise IndexError("Index must be tuple of two values")
        if not isinstance(key[0], (int, slice)) or not isinstance(key[1], (int, slice)):
            raise IndexError("Index must be tuple of int/slice values")
        if not isinstance(value, self._inner_type):
            raise ValueError(f"Value must be type {self._inner_type}")
        # Bounds
        if (isinstance(key[0], int) and (key[0] < 0 or key[0] >= self._width)) or \
                (isinstance(key[1], int) and (key[1] < 0 or key[1] >= self._height)):
            # (isinstance(index[0], slice) and (index[0] and)):
            raise IndexError("Matrix doesn't support negative or overflow indexes")

        if isinstance(key[0], int) and isinstance(key[1], int):
            self._values[key[1]][key[0]] = value
        raise NotImplementedError()

    def transpose(self) -> None:
        self._values = [[self._values[j][i] for j in range(len(self._values))] for i in range(len(self._values[0]))]
        # self._values = list(map(list, zip(*self._values)))
        self._height, self._width = self._width, self._height

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
        if not isinstance(values, list):
            raise ValueError()
        h = len(values)
        if h == 0:
            raise ValueError('Cannot create matrix from empty list')
        w = len(values[0])
        if w == 0:
            raise ValueError('Cannot create matrix with width = 0')
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
    def zero_matrix(cls, size, *, boolean_matrix: bool = False) -> Matrix[_MVT]:
        if boolean_matrix:
            return cls.generate(width=size, height=size, value=False)
        return cls.generate(width=size, height=size, value=0)

    @classmethod
    def identity(cls, size, *, boolean_matrix: bool = False) -> Matrix[_MVT]:
        if boolean_matrix:
            return cls.generate(width=size, height=size, value=lambda x, y: x==y)
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
            raise NotSquareMatrix()
        return [self._values[i][i] for i in range(self.width)]

    @property
    def trace(self) -> Union[int, Any]:
        return sum(self.main_diagonal)

    def _minor(self, i, j):
        return Matrix(self.width-1, self.height-1,
                      [row[:j] + row[j + 1:] for row in (self._values[:i] + self._values[i + 1:])])

    @property
    def determinant(self) -> float:
        # if not self.is_square:
        #     raise NotSquareMatrix()
        m = [row[:] for row in self._values]
        for fd in range(self.width):
            for i in range(fd+1, self.width):
                if m[fd][fd] == 0:
                    m[fd][fd] = 1.0e-18
                s = m[i][fd] / m[fd][fd]
                for j in range(self.width):
                    m[i][j] = m[i][j] - s*m[fd][j]
        p = 1.0
        for i in range(self.width):
            p *= m[i][i]
        return round(p, 10)
        # if self.width == 2:
        #     return self._values[0][0] * self._values[1][1] - self._values[0][1] * self._values[1][0]
        #
        # determinant = 0
        # for c in range(self.width):
        #     determinant += ((-1) ** c) * self._values[0][c] * self.determinant(self._minor(0, c))
        # return determinant


    def __contains__(self, item: Matrix[_MVT] or List[List[_MVT]]) -> bool:
        other = Matrix.from_nested_list(item) if isinstance(item, List) else item
        if other.width > self.width or other.height > self.height:
            return False
        offsets_x = range(self.width - other.width + 1)
        offsets_y = range(self.height - other.height + 1)
        for offset_y in offsets_y:
            for offset_x in offsets_x:
                eq = True
                for y in range(other.height):
                    for x in range(other.width):
                        if self._values[y+offset_y][x+offset_x] != other._values[y][x]:
                            eq = False
                            break
                    if not eq:
                        break
                if eq:
                    return True
        return False

    def __invert__(self):
        """ Overrides operator ~A """
        # raise NotImplementedError()
        # if not self.is_square:
        #     raise NotSquareMatrix()
        d = self.determinant
        if self.width == 2:
            return Matrix.from_nested_list([[self._values[1][1] / d, -1 * self._values[0][1] / d],
                    [-1 * self._values[1][0] / d, self._values[0][0] / d]])
        else:
            raise NotImplementedError('Inverse matrix for > 2x2 is not supported yet')

    def __add__(self, other: Matrix) -> Matrix[_MVT]:
        if not isinstance(other, Matrix):
            raise AttributeError()
        if self.size != other.size:
            raise AttributeError('Invalid matrix size')

        res = [[self._values[j][i] + other._values[j][i] for i in range(self.width)] for j in range(self.height)]
        return Matrix.from_nested_list(res)

    def __sub__(self, other: Matrix) -> Matrix[_MVT]:
        if not isinstance(other, Matrix):
            raise AttributeError()
        if self.size != other.size:
            raise AttributeError('Invalid matrix size')

        res = [[
            self._values[j][i] and not other._values[j][i]
            if (isinstance(self._values[j][i], bool) and isinstance(other._values[j][i], bool))
            else self._values[j][i] - other._values[j][i]
            for i in range(self.width)]
            for j in range(self.height)
        ]
        return Matrix.from_nested_list(res)

    def __itruediv__(self, other: Union[int, float]) -> Matrix[_MVT]:
        if not isinstance(other, numbers.Number):
            raise AttributeError()

        for i in range(self.width):
            for j in range(self.height):
                self._values[i][j] /= other

        return self

    def __imul__(self, other: Union[Matrix, int, float]) -> Matrix[_MVT]:
        if isinstance(other, Matrix):
            c = []
            for i in range(0, self.width):
                temp = []
                for j in range(0, other.height):
                    s = 0
                    for k in range(0, self.height):
                        s += self._values[i][k] * other._values[k][j]
                    temp.append(s)
                c.append(temp)
            self._values = c
            self._height = other.height
            return self

        if not isinstance(other, numbers.Number):
            raise AttributeError()

        for i in range(self.width):
            for j in range(self.height):
                self._values[i][j] *= other

        return self

    def __mul__(self, other: Union[Matrix, _MVT]) -> Matrix[_MVT]:
        if isinstance(other, numbers.Number):
            return Matrix.from_nested_list([[elem * other for elem in row] for row in self._values])

        if not isinstance(other, Matrix):
            raise AttributeError()

        if self.width != other.height:
            raise AttributeError('Incorrect matrix size for multiplication')

        c = []
        for i in range(0, self.width):
            temp = []
            for j in range(0, other.height):
                s = 0
                for k in range(0, self.height):
                    s += self._values[i][k] * other._values[k][j]
                temp.append(s)
            c.append(temp)
        return Matrix(self.width, other.width, c)

    def __eq__(self, other: List[List[_MVT]] or Matrix[_MVT] or bool) -> bool:
        if isinstance(other, Matrix):
            return self.size == other.size and self._values == other._values
        if isinstance(other, bool):
            return all([all([elem is other for elem in row]) for row in self._values])
        else:
            try:
                return self._values == other
            except ValueError:
                return False

    @property
    def rotated_counterclockwise(self):
        return Matrix(self._height, self._width, [[self._values[j][i] for j in range(self.height)] for i in range(self.width - 1, -1, -1)])

    @property
    def rotated_clockwise(self):
        return Matrix(self._height, self._width, [[self._values[j][i] for j in range(self.height - 1, -1, -1)] for i in range(0, self.width)])

    @property
    def mirrored_horizontaly(self):
        return Matrix(self._width, self.height, [row[::-1] for row in self._values])

    @property
    def mirrored_verticaly(self):
        return Matrix(self._width, self.height, [row[::] for row in self._values[::-1]])

    def __and__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix):
            raise AttributeError()
        if self.size != other.size:
            raise AttributeError('Invalid matrix size')

        res = [[self._values[j][i] & other._values[j][i] for i in range(self.width)] for j in range(self.height)]
        return Matrix.from_nested_list(res)

    def __or__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix):
            raise AttributeError()
        if self.size != other.size:
            raise AttributeError('Invalid matrix size')

        res = [[self._values[j][i] | other._values[j][i] for i in range(self.width)] for j in range(self.height)]
        return Matrix.from_nested_list(res)

    def __xor__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix):
            raise AttributeError()
        if self.size != other.size:
            raise AttributeError('Invalid matrix size')

        res = [[self._values[j][i] ^ other._values[j][i] for i in range(self.width)] for j in range(self.height)]
        return BitMatrix.from_nested_list(res)

    def __neg__(self):
        res = [[not self._values[j][i] if (self._values[j][i],bool) else -self._values[j][i] for i in range(self.width)] for j in range(self.height)]
        return BitMatrix.from_nested_list(res)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self._values}"


class BitMatrix(Matrix[bool]):
    def zero_matrix(cls, size) -> Matrix[_MVT]:
        return cls.generate(size, size, False)

    def identity(cls, size) -> Matrix[_MVT]:
        return cls.generate(size, size, lambda x,y: x==y)