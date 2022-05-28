from __future__ import annotations

import numbers
from collections.abc import Callable, Iterator
from enum import Enum, unique, auto, IntEnum
from typing import Tuple, Union, List, Generic, TypeVar, Optional, Any

# Matrix Value Type & Matrix Key Type
_MVT = TypeVar("_MVT")  # Union[int, float, complex] # + maybe bool
_MKT = TypeVar("_MKT", Tuple[int, int], Tuple[int, slice], Tuple[slice, int], Tuple[slice, slice])


class NotSquareMatrix(ValueError):
    pass


class Matrix(Generic[_MVT]):
    # ======== Class logic ========
    def __init__(self, width: int, height: int, values: List[List[_MVT]]):
        self._width: int = width
        self._height: int = height
        self._values: List[List[_MVT]] = values

    def __getitem__(
        self,
        index: _MKT,
    ) -> _MVT:
        # Type Check
        if not isinstance(index, tuple):
            raise IndexError("Index must be tuple")
        if len(index) != 2:
            raise IndexError("Index must be tuple of two values")
        if not isinstance(index[0], (int, slice)) or not isinstance(index[1], (int, slice)):
            raise IndexError("Index must be tuple of int/slice values")
        # Bounds
        if isinstance(index[0], int) and (index[0] < 0 or index[0] >= self._width):
            raise IndexError("Matrix doesn't support negative or overflow indexes")
        if isinstance(index[1], int) and (index[1] < 0 or index[1] >= self._height):
            raise IndexError("Matrix doesn't support negative or overflow indexes")

        if isinstance(index[1], int):
            if isinstance(index[0], slice):
                return self._values[index[1]][index[0]][:]
            return self._values[index[1]][index[0]]
        return [row[index[0]] for row in self._values[index[1]]]

    def __setitem__(
        self,
        key: _MKT,
        value: _MVT,
    ) -> None:
        # Type Check
        if not isinstance(key, tuple):
            raise IndexError("Index must be tuple")
        if len(key) != 2:
            raise IndexError("Index must be tuple of two values")
        if not isinstance(key[0], (int, slice)) or not isinstance(key[1], (int, slice)):
            raise IndexError("Index must be tuple of int/slice values")
        # Bounds
        if isinstance(key[0], int) and (key[0] < 0 or key[0] >= self._width):
            raise IndexError("Matrix doesn't support negative or overflow indexes")
        if isinstance(key[1], int) and (key[1] < 0 or key[1] >= self._height):
            raise IndexError("Matrix doesn't support negative or overflow indexes")

        if isinstance(key[0], int) and isinstance(key[1], int):
            self._values[key[1]][key[0]] = value
        raise NotImplementedError()

    def __contains__(self, item: Matrix[_MVT] or List[List[_MVT]]) -> bool:
        other = Matrix.from_nested_list(item) if isinstance(item, List) else item
        if other.width > self.width or other.height > self.height:
            return False
        for j in range(self.height - other.height + 1):
            for i in range(self.width - other.width + 1):
                if self[i : i + other.width, j : j + other.height] == other:
                    return True
        return False

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def size(self) -> Tuple[int, int]:
        return self._width, self._height

    def __repr__(self):
        return f"<{self.__class__.__name__}({self._values})>"

    # TODO:
    class Walkthrow(Enum):
        DEFAULT = 0
        SNAKE = 1
        SPIRAL = 2

    # ======== Factories ========

    @classmethod
    def generate(
        cls,
        width,
        height,
        value: Union[Callable[[int, int], _MVT], Callable[[], _MVT], _MVT, Iterator],
        *,
        walkthrow: Walkthrow = Walkthrow.DEFAULT,
    ) -> Matrix[_MVT]:
        """Generates matrix from size and generator, for example (2, 2, lambda x,y: x+y"""
        values = []
        for j in range(height):
            row = []
            for i in range(width):
                if callable(value):
                    if value.__code__.co_argcount == 2:  # noqa
                        row.append(value(i, j))
                    elif value.__code__.co_argcount == 0:  # noqa
                        row.append(value())
                    else:
                        raise ValueError("Incorrect number of arguments for generator")
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
        height = len(values)
        if height == 0:
            raise ValueError("Cannot create matrix from empty list")
        width = len(values[0])
        if width == 0:
            raise ValueError("Cannot create matrix with width = 0")
        if not all(len(row) == width for row in values):
            raise ValueError("All rows must have equal length")
        return Matrix(width, height, values)

    @classmethod
    def from_joined_lists(cls, width: int, height: int = None, *, values: List[_MVT] or range) -> Matrix[_MVT]:
        lists = []
        if isinstance(values, range):
            values = list(values)
        for i in range(0, len(values), width):
            lists.append(values[i : i + width])
        if height and len(lists) != height or len(lists[-1]) != width:
            raise ValueError("Incorrect elements count")
        return cls.from_nested_list(lists)

    @classmethod
    def from_lists(cls, *lists: List[_MVT]) -> Matrix[_MVT]:
        return cls.from_nested_list(values=list(lists))

    @classmethod
    def zero_matrix(cls, size: Union[int, Tuple[int, int]], *, boolean_matrix: bool = False) -> Matrix[_MVT]:
        if isinstance(size, int):
            return cls.generate(width=size, height=size, value=False if boolean_matrix else 0)
        return cls.generate(width=size[0], height=size[1], value=False if boolean_matrix else 0)

    @classmethod
    def identity(cls, size: int, *, boolean_matrix: bool = False) -> Matrix[_MVT]:
        if boolean_matrix:
            return cls.generate(width=size, height=size, value=lambda x, y: x == y)
        return cls.generate(width=size, height=size, value=lambda x, y: 1 if x == y else 0)

    @classmethod
    def input_matrix(
        cls,
        height: Optional[int] = None,
        width: Optional[int] = None,
        postprocess: Callable[[str]:_MVT] = lambda x: int(x),
        *,
        width_first: bool = False,
        walkthrow: Walkthrow = Walkthrow.DEFAULT,
    ) -> Matrix[_MVT]:
        if width_first:
            height = height or int(input())
            width = width or int(input())
        else:
            width = width or int(input())
            height = height or int(input())
        assert isinstance(width, int)
        assert isinstance(height, int)
        return cls.generate(width, height, lambda: postprocess(input()), walkthrow=walkthrow)

    # ======== Matrix special logic ========

    def transpose(self) -> None:
        self._values = [[self._values[j][i] for j in range(len(self._values))] for i in range(len(self._values[0]))]
        self._height, self._width = self._width, self._height

    @property
    def is_square(self) -> bool:
        return self._width == self._height

    @property
    def is_zero(self) -> bool:
        raise NotImplementedError()

    @property
    def is_identity(self) -> bool:
        raise NotImplementedError()

    @property
    def main_diagonal(self) -> List[_MVT]:
        """Returns list of main diagonal elements"""
        if not self.is_square:
            raise NotSquareMatrix()
        return [self._values[i][i] for i in range(self.width)]

    @property
    def trace(self) -> _MVT:
        return sum(self.main_diagonal)

    def get_minor(self, i, j):
        return Matrix(
            self.width - 1,
            self.height - 1,
            [row[:j] + row[j + 1 :] for row in (self._values[:i] + self._values[i + 1 :])],
        )

    @property
    def determinant(self) -> float:
        # if not self.is_square:
        #     raise NotSquareMatrix()
        matrix = [row[:] for row in self._values]
        for fd in range(self.width):
            for i in range(fd + 1, self.width):
                if matrix[fd][fd] == 0:
                    matrix[fd][fd] = 1.0e-18
                s = matrix[i][fd] / matrix[fd][fd]
                for j in range(self.width):
                    matrix[i][j] = matrix[i][j] - s * matrix[fd][j]
        p = 1.0
        for i in range(self.width):
            p *= matrix[i][i]
        return round(p, 10)
        # if self.width == 2:
        #     return self._values[0][0] * self._values[1][1] - self._values[0][1] * self._values[1][0]
        #
        # determinant = 0
        # for c in range(self.width):
        #     determinant += ((-1) ** c) * self._values[0][c] * self.determinant(self._minor(0, c))
        # return determinant

    # ======== Math operations on matrix ========

    def __invert__(self):
        """Overrides operator ~A"""
        # raise NotImplementedError()
        # if not self.is_square:
        #     raise NotSquareMatrix()
        det = self.determinant
        if self.width == 2:
            return Matrix.from_nested_list(
                [
                    [self._values[1][1] / det, -1 * self._values[0][1] / det],
                    [-1 * self._values[1][0] / det, self._values[0][0] / det],
                ]
            )
        else:
            raise NotImplementedError("Inverse matrix for > 2x2 is not supported yet")

    def __base_binary_operation_creating_new_entity__(
        self,
        other: Matrix[_MVT] = None,
        operation: Callable[[_MVT, _MVT], _MVT] = None,
    ) -> Matrix[_MVT]:
        # Check if operation is defined
        assert operation is not None
        # Check argument
        if not isinstance(other, Matrix):
            raise AttributeError("Other argument must be a matrix")
        if self.size != other.size:
            raise AttributeError("Invalid matrix size")
        # Calculating the result
        return Matrix(
            width=self.width,
            height=self.height,
            values=[
                [operation(self._values[j][i], other._values[j][i]) for i in range(self.width)]
                for j in range(self.height)
            ],
        )

    def __base_binary_operation_applying_to_self__(
        self,
        other: Matrix[_MVT] = None,
        operation: Callable[[_MVT, _MVT], _MVT] = None,
    ) -> Matrix[_MVT]:
        # Check if operation is defined
        assert operation is not None
        # Check argument
        if not isinstance(other, Matrix):
            raise AttributeError("Other argument must be a matrix")
        if self.size != other.size:
            raise AttributeError("Invalid matrix size")
        for i in range(self.width):
            for j in range(self.height):
                self._values[j][i] = operation(self._values[j][i], other._values[j][i])
        return self

    def __add__(self, other: Matrix[_MVT]) -> Matrix[_MVT]:
        return self.__base_binary_operation_creating_new_entity__(other, lambda x, y: x + y)

    def __iadd__(self, other: Matrix[_MVT]) -> Matrix[_MVT]:
        return self.__base_binary_operation_applying_to_self__(other, lambda x, y: x + y)

    def __sub__(self, other: Matrix) -> Matrix[_MVT]:
        return self.__base_binary_operation_creating_new_entity__(
            other, lambda x, y: x and not y if isinstance(x, bool) and isinstance(y, bool) else x - y
        )

    def __isub__(self, other: Matrix) -> Matrix[_MVT]:
        return self.__base_binary_operation_applying_to_self__(
            other, lambda x, y: x and not y if isinstance(x, bool) and isinstance(y, bool) else x - y
        )

    def __itruediv__(self, other: Union[int, float]) -> Matrix[_MVT]:
        if not isinstance(other, numbers.Number):
            raise AttributeError()

        for i in range(self.width):
            for j in range(self.height):
                self._values[i][j] /= other

        return self

    def __imul__(self, other: Union[Matrix, int, float]) -> Matrix[_MVT]:
        if isinstance(other, Matrix):
            new_values = []
            for i in range(0, self.width):
                row = []
                for j in range(0, other.height):
                    row.append(sum([self._values[i][k] * other._values[k][j] for k in range(self.height)]))
                new_values.append(row)
            self._values = new_values
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
            raise AttributeError("Incorrect matrix size for multiplication")

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
            return self._values == other

    @property
    def rotated_counterclockwise(self):
        return Matrix(
            self._height,
            self._width,
            [[self._values[j][i] for j in range(self.height)] for i in range(self.width - 1, -1, -1)],
        )

    @property
    def rotated_clockwise(self):
        return Matrix(
            self._height,
            self._width,
            [[self._values[j][i] for j in range(self.height - 1, -1, -1)] for i in range(0, self.width)],
        )

    @property
    def mirrored_horizontaly(self):
        return Matrix(self._width, self.height, [row[::-1] for row in self._values])

    @property
    def mirrored_verticaly(self):
        return Matrix(self._width, self.height, [row[::] for row in self._values[::-1]])

    # ======== Boolean logic operations on matrix ========

    def __base_boolean_operation_creating_new_entity__(
        self,
        other: Matrix[bool] = None,
        operation: Callable[[bool, bool], bool] = None,
    ) -> Matrix[bool]:
        return self.__base_binary_operation_creating_new_entity__(other, operation)

    def __base_boolean_operation_applying_to_self__(
        self,
        other: Matrix[bool] = None,
        operation: Callable[[bool, bool], bool] = None,
    ) -> Matrix[bool]:
        return self.__base_binary_operation_applying_to_self__(other, operation)

    def __and__(self, other: Matrix[bool]) -> Matrix:
        return self.__base_boolean_operation_creating_new_entity__(other, lambda x, y: x & y)

    def __or__(self, other: Matrix[bool]) -> Matrix:
        return self.__base_boolean_operation_creating_new_entity__(other, lambda x, y: x | y)

    def __xor__(self, other: Matrix[bool]) -> Matrix:
        return self.__base_boolean_operation_creating_new_entity__(other, lambda x, y: x ^ y)

    def __iand__(self, other: Matrix[bool]) -> Matrix:
        return self.__base_boolean_operation_applying_to_self__(other, lambda x, y: x & y)

    def __ior__(self, other: Matrix[bool]) -> Matrix:
        return self.__base_boolean_operation_applying_to_self__(other, lambda x, y: x | y)

    def __ixor__(self, other: Matrix[bool]) -> Matrix:
        return self.__base_boolean_operation_applying_to_self__(other, lambda x, y: x ^ y)

    def __neg__(self):
        return [
            [
                not self._values[j][i] if isinstance(self._values[j][i], bool) else -self._values[j][i]
                for i in range(self.width)
            ]
            for j in range(self.height)
        ]
