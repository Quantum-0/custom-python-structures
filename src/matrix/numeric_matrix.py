from __future__ import annotations
from typing import TypeVar, Union, Tuple, List

from src.matrix.matrix import Matrix

_MVT = TypeVar("_MVT", float, int)  # complex
Self = TypeVar("Self", bound="NumericMatrix")


class NumericMatrix(Matrix[_MVT]):
    _values: List[List[_MVT]]

    @classmethod
    def zero_matrix(cls, size: Union[int, Tuple[int, int]]) -> Matrix[int]:
        if isinstance(size, int):
            return NumericMatrix(width=size, height=size, values=[list([0] * size)] * size)
        return NumericMatrix(width=size[0], height=size[1], values=[list([0] * size[0])] * size[1])

    @classmethod
    def identity(cls, size: int) -> Matrix[int]:
        return cls.generate(width=size, height=size, value=lambda x, y: 1 if x == y else 0)

    @property
    def is_zero(self) -> bool:
        return all(all(elem == 0 for elem in row) for row in self._values)

    @property
    def is_identity(self) -> bool:
        return all(all(val == (1 if i == j else 0) for i, val in enumerate(row)) for j, row in enumerate(self._values))

    @property
    def trace(self) -> _MVT:
        return sum(self.main_diagonal)

    @property
    def determinant(self) -> float:
        # pylint: disable=invalid-name
        # if not self.is_square:
        #     raise NotSquareMatrix()
        matrix = [row[:] for row in self._values]
        for fd in range(self.width):
            for i in range(fd + 1, self.width):
                if matrix[fd][fd] == 0:
                    matrix[fd][fd] = 1.0e-18  # type: ignore
                s = matrix[i][fd] / matrix[fd][fd]
                for j in range(self.width):
                    matrix[i][j] = matrix[i][j] - s * matrix[fd][j]  # type: ignore
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

    def __invert__(self) -> NumericMatrix[float]:
        det = self.determinant
        if self.width > 2:
            raise NotImplementedError("Inverse matrix for > 2x2 is not supported yet")
        return NumericMatrix(
            width=2,
            height=2,
            values=[
                [self._values[1][1] / det, -1 * self._values[0][1] / det],
                [-1 * self._values[1][0] / det, self._values[0][0] / det],
            ],
        )

    def __add__(self, other: Matrix[_MVT]) -> NumericMatrix[_MVT]:
        return self.__base_binary_operation_creating_new_entity__(other, lambda x, y: x + y)

    def __iadd__(self, other: Matrix[_MVT]) -> NumericMatrix[_MVT]:
        return self.__base_binary_operation_applying_to_self__(other, lambda x, y: x + y)

    def __sub__(self, other: Matrix) -> NumericMatrix[_MVT]:
        return self.__base_binary_operation_creating_new_entity__(other, lambda x, y: x - y)

    def __isub__(self, other: Matrix) -> NumericMatrix[_MVT]:
        return self.__base_binary_operation_applying_to_self__(other, lambda x, y: x - y)

    def __itruediv__(self, other: Union[int, float]) -> NumericMatrix[_MVT]:
        if not isinstance(other, (int, float, complex)):
            raise AttributeError()

        for i in range(self.width):
            for j in range(self.height):
                self._values[i][j] /= other  # type: ignore

        return self

    def __imul__(self, other: Union[NumericMatrix, int, float]) -> NumericMatrix[_MVT]:
        if isinstance(other, NumericMatrix):
            new_values = []
            for i in range(0, self.width):
                row = []
                for j in range(0, other.height):
                    row.append(sum([self._values[i][k] * other._values[k][j] for k in range(self.height)]))
                new_values.append(row)
            self._values = new_values
            self._height = other.height
            return self

        if not isinstance(other, (int, float, complex)):
            raise AttributeError()

        for j in range(self.height):
            for i in range(self.width):
                self._values[i][j] *= other  # type: ignore

        return self

    def __mul__(self, other: Union[NumericMatrix, _MVT]) -> NumericMatrix[_MVT]:
        if isinstance(other, (int, float)):
            return NumericMatrix(self._width, self._height, [[elem * other for elem in row] for row in self._values])

        if not isinstance(other, Matrix):
            raise AttributeError()

        if self.width != other.height:
            raise AttributeError("Incorrect matrix size for multiplication")

        new_values = []
        for i in range(0, self.width):
            row = []
            for j in range(0, other.height):
                row.append(sum([self._values[i][k] * other._values[k][j] for k in range(self.height)]))
            new_values.append(row)
        return self.__class__(self.width, other.width, new_values)

    def __neg__(self) -> NumericMatrix[_MVT]:
        return NumericMatrix(
            width=self.width,
            height=self.height,
            values=[[-self._values[j][i] for i in range(self.width)] for j in range(self.height)],
        )
