from __future__ import annotations
from abc import ABC
from collections.abc import Iterable
from typing import Type, ClassVar

from src.matrix.matrix import Matrix, _MVT


class MatrixIterator(Iterable, ABC):
    WALKTHROW_TYPE: ClassVar[Matrix.Walkthrow]

    @classmethod
    def iterate(cls, matrix: Matrix, iteration_type: Matrix.Walkthrow):
        iter_type: Type[MatrixIterator] = cls._get_iterator_type(walkthrow_type=iteration_type)
        return iter_type(matrix)

    def __init__(self, matrix: Matrix):
        self._ptr = 0
        self._matrix = matrix
        self._len = self._matrix.width * self._matrix.height

    def __iter__(self) -> MatrixIterator:
        return self

    def __next__(self) -> None:
        self._ptr += 1
        if self._ptr > (self._matrix.width * self._matrix.height):
            raise StopIteration()

    @classmethod
    def _get_iterator_type(cls, walkthrow_type: Matrix.Walkthrow) -> Type[MatrixIterator]:
        found = [iterator for iterator in cls.__subclasses__() if iterator.WALKTHROW_TYPE == walkthrow_type]
        assert len(found) < 2, f"Duplicate implementation for iterator {walkthrow_type}: {found}"
        assert len(found) == 1, f"Cannot find implementation for iterator {walkthrow_type}"
        return found[0]


class DefaultMatrixIterator(MatrixIterator):
    WALKTHROW_TYPE = Matrix.Walkthrow.DEFAULT

    def __next__(self) -> _MVT:
        if self._ptr < self._len:
            value = self._matrix[self._ptr % self._matrix.width, self._ptr // self._matrix.width]
        super().__next__()
        return value


class ResersedMatrixIterator(MatrixIterator):
    WALKTHROW_TYPE = Matrix.Walkthrow.REVERSED

    def __next__(self) -> _MVT:
        if self._ptr < self._len:
            value = self._matrix[
                (self._len - 1 - self._ptr) % self._matrix.width, (self._len - 1 - self._ptr) // self._matrix.width
            ]
            self._ptr += 1
            return value
        raise StopIteration


class SnakeMatrixIterator(MatrixIterator):
    WALKTHROW_TYPE = Matrix.Walkthrow.SNAKE

    def __next__(self) -> _MVT:
        if self._ptr < self._len:
            j = self._ptr // self._matrix.width
            i = (self._len - 1 - self._ptr) % self._matrix.width if j % 2 == 1 else self._ptr % self._matrix.width
            value = self._matrix[i, j]
            self._ptr += 1
            return value
        raise StopIteration


class SpiralMatrixIterator(MatrixIterator):
    WALKTHROW_TYPE = Matrix.Walkthrow.SPIRAL

    def __next__(self) -> _MVT:
        raise NotImplementedError


class RowMatrixIterator(MatrixIterator):
    WALKTHROW_TYPE = Matrix.Walkthrow.ROWS

    def __next__(self) -> list:  # type: ignore
        if self._ptr >= self._matrix.height:
            raise StopIteration
        value = self._matrix[:, self._ptr]
        self._ptr += 1
        return value


class ColumnMatrixIterator(MatrixIterator):
    WALKTHROW_TYPE = Matrix.Walkthrow.COLUMNS

    def __next__(self) -> list:  # type: ignore
        if self._ptr == self._matrix.width:
            raise StopIteration
        value = self._matrix[self._ptr, :]
        self._ptr += 1
        return value
