from __future__ import annotations

from collections.abc import Callable, Iterator
from enum import Enum
from typing import Tuple, Union, List, Generic, TypeVar, Optional

# Matrix Value Type & Matrix Key Type
_MVT = TypeVar("_MVT")
_MKT = TypeVar("_MKT", Tuple[int, int], Tuple[int, slice], Tuple[slice, int], Tuple[slice, slice])


class NotSquareMatrix(ValueError):
    pass


class Matrix(Generic[_MVT]):
    def __init__(self, width: int, height: int, values: List[List[_MVT]]):
        self._width: int = width
        self._height: int = height
        self._values: List[List[_MVT]] = values

    def __getitem__(
        self,
        index: _MKT,
    ) -> Union[_MVT, List[_MVT], List[List[_MVT]]]:
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
            raise IndexError(f"Matrix doesn't support negative or overflow indexes: {index}")

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

        # Check and recreate slices
        if isinstance(key[0], slice):
            if key[0].step is not None:
                raise NotImplementedError
            key = slice(key[0].start or 0, key[0].stop or self._width, None), key[1]
        if isinstance(key[1], slice):
            if key[1].step is not None:
                raise NotImplementedError
            key = key[0], slice(key[1].start or 0, key[1].stop or self._height, None)

        if isinstance(key[0], int) and isinstance(key[1], int):
            self._values[key[1]][key[0]] = value
        elif isinstance(key[0], slice) and isinstance(key[1], int):
            if not isinstance(value, list):
                raise ValueError
            if key[0].stop - key[0].start != len(value):
                raise ValueError
            self._values[key[1]][key[0]] = value
        elif isinstance(key[0], int) and isinstance(key[1], slice):
            if not isinstance(value, list):
                raise ValueError
            if key[1].stop - key[1].start != len(value):
                raise ValueError
            for j, val in zip(range(key[1].start, key[1].stop), value):
                self._values[j][key[0]] = val
        elif isinstance(key[0], slice) and isinstance(key[1], slice):
            if not isinstance(value, list):
                raise ValueError
            if not all(isinstance(val, list) for val in value):
                raise ValueError
            if key[1].stop - key[1].start != len(value):
                raise ValueError
            if not all(len(val) == key[0].stop - key[0].start for val in value):
                raise ValueError
            for j, val in zip(range(key[1].start, key[1].stop), value):
                self._values[j][key[0]] = val

    def __contains__(self, item: Union[Matrix[_MVT], List[List[_MVT]]]) -> bool:
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

    class Walkthrow(Enum):
        DEFAULT = 0
        REVERSED = 1
        SNAKE = 2
        SPIRAL = 3
        COLUMNS = 4
        ROWS = 5

    # ======== Factories ========

    @classmethod
    def generate(
        cls,
        width,
        height,
        value: Union[Callable[[int, int], _MVT], Callable[[int], _MVT], Callable[[], _MVT], _MVT, Iterator],
        *,
        by_rows: bool = False,
        walkthrow: Walkthrow = Walkthrow.DEFAULT,  # type: ignore # pylint: disable=W0613 # TODO
    ):
        """Generates matrix from size and generator, for example (2, 2, lambda x,y: x+y"""
        if not isinstance(width, int) or not isinstance(height, int):
            raise TypeError
        if width < 1 or height < 1:
            raise ValueError

        values = []
        for j in range(height):
            if not by_rows:
                row = []
                for i in range(width):
                    if callable(value):
                        if value.__code__.co_argcount == 2:  # noqa
                            row.append(value(i, j))  # type: ignore
                        elif value.__code__.co_argcount == 0:  # noqa
                            row.append(value())  # type: ignore
                        else:
                            raise ValueError("Incorrect number of arguments for generator")
                    elif isinstance(value, Iterator):
                        row.append(next(value))
                    else:
                        row.append(value)
            else:
                if callable(value):
                    if value.__code__.co_argcount == 1:  # noqa
                        row = list(value(j))  # type: ignore
                    elif value.__code__.co_argcount == 0:  # noqa
                        row = list(value())  # type: ignore
                    else:
                        raise ValueError("Incorrect number of arguments for generator")
                elif isinstance(value, Iterator):
                    row = list(next(value))
                else:
                    row = list(value)
            values.append(row)
        return cls(width=width, height=height, values=values)

    @classmethod
    def from_nested_list(cls, values: List[List[_MVT]]) -> Matrix[_MVT]:
        if not isinstance(values, list):
            raise TypeError
        height = len(values)
        if height == 0:
            raise ValueError("Cannot create matrix from empty list")
        width = len(values[0])
        if width == 0:
            raise ValueError("Cannot create matrix with width = 0")
        if not all(len(row) == width for row in values):
            raise ValueError("All rows must have equal length")
        return cls(width, height, values)

    @classmethod
    def from_joined_lists(cls, width: int, height: int = None, *, values: Union[List[_MVT], range]) -> Matrix:
        lists = []
        if isinstance(values, range):
            values = list(values)  # type: ignore # FIXME
        for i in range(0, len(values), width):
            lists.append(values[i : i + width])
        if height and len(lists) != height or len(lists[-1]) != width:
            raise ValueError("Incorrect elements count")
        return cls.from_nested_list(lists)

    @classmethod
    def from_lists(cls, *lists: List[_MVT]) -> Matrix:
        return cls.from_nested_list(values=list(lists))

    @classmethod
    def input_matrix(
        cls,
        height: Optional[int] = None,
        width: Optional[int] = None,
        postprocess: Union[Callable[[str], _MVT], Callable[[str], List[_MVT]]] = None,  # noqa
        *,
        width_first: bool = False,
        by_rows: bool = False,
        walkthrow: Walkthrow = Walkthrow.DEFAULT,
    ) -> Matrix:  # pragma: no cover
        if postprocess is None:
            if by_rows:
                postprocess = lambda x: list(map(int, x.split()))
            else:
                postprocess = int.__call__
        assert postprocess is not None
        if width_first:
            height = height or int(input())
            width = width or int(input())
        else:
            width = width or int(input())
            height = height or int(input())
        assert isinstance(width, int)
        assert isinstance(height, int)
        return cls.generate(width, height, lambda: postprocess(input()), by_rows=by_rows, walkthrow=walkthrow)

    def transpose(self) -> None:
        self._values = [[self._values[j][i] for j in range(len(self._values))] for i in range(len(self._values[0]))]
        self._height, self._width = self._width, self._height

    @property
    def is_square(self) -> bool:
        return self._width == self._height

    @property
    def main_diagonal(self) -> List[_MVT]:
        """Returns list of main diagonal elements"""
        if not self.is_square:
            raise NotSquareMatrix()
        return [self._values[i][i] for i in range(self.width)]

    def get_minor(self, i: int, j: int):
        if not isinstance(i, int) or not isinstance(j, int):
            raise TypeError
        if not 0 <= i < self._width or not 0 <= j < self._height:
            raise ValueError
        return Matrix(
            self.width - 1,
            self.height - 1,
            [row[:j] + row[j + 1 :] for row in self._values[:i] + self._values[i + 1 :]],
        )

    def __eq__(self, other: Union[List[List[_MVT]], Matrix[_MVT], object]) -> bool:
        if isinstance(other, Matrix):
            return self.size == other.size and self._values == other._values
        if isinstance(other, list) and all(isinstance(l, list) for l in other):
            return self._values == other
        return False

    @property
    def rotated_counterclockwise(self) -> Matrix:
        return self.__class__(
            self._height,
            self._width,
            [[self._values[j][i] for j in range(self._height)] for i in range(self._width - 1, -1, -1)],
        )

    @property
    def rotated_clockwise(self) -> Matrix:
        return self.__class__(
            self._height,
            self._width,
            [[self._values[j][i] for j in range(self._height - 1, -1, -1)] for i in range(0, self._width)],
        )

    @property
    def mirrored_horizontaly(self) -> Matrix:
        return self.__class__(self._width, self._height, [row[::-1] for row in self._values])

    @property
    def mirrored_verticaly(self) -> Matrix:
        return self.__class__(self._width, self._height, [row[::] for row in self._values[::-1]])

    def __base_binary_operation_creating_new_entity__(
        self,
        other: Matrix[_MVT],
        operation: Callable[[_MVT, _MVT], _MVT],
    ):
        # Check if operation is defined
        assert operation is not None
        # Check argument
        if not isinstance(other, Matrix):
            raise AttributeError("Other argument must be a matrix")
        if self.size != other.size:
            raise AttributeError("Invalid matrix size")
        # Calculating the result
        return self.__class__(
            width=self.width,
            height=self.height,
            values=[
                # type: ignore # pylint: disable=protected-access
                [operation(self._values[j][i], other._values[j][i]) for i in range(self.width)]
                for j in range(self.height)
            ],
        )

    def __base_binary_operation_applying_to_self__(
        self,
        other: Matrix[_MVT],
        operation: Callable[[_MVT, _MVT], _MVT],
    ):
        # Check if operation is defined
        assert operation is not None
        # Check argument
        if not isinstance(other, Matrix):
            raise AttributeError("Other argument must be a matrix")
        if self.size != other.size:
            raise AttributeError("Invalid matrix size")
        for i in range(self.width):
            for j in range(self.height):
                # type: ignore # pylint: disable=protected-access
                self._values[j][i] = operation(self._values[j][i], other._values[j][i])
        return self
