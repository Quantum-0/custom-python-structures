from __future__ import annotations
from typing import Union, Tuple, List, Any, TypeVar

from src.matrix.matrix import Matrix


Self = TypeVar("Self", bound="NumericMatrix")


class BitMatrix(Matrix[bool]):
    @classmethod
    def zero_matrix(cls, size: Union[int, Tuple[int, int]]) -> BitMatrix:
        if isinstance(size, int):
            return cls.generate(width=size, height=size, value=False)
        return cls.generate(width=size[0], height=size[1], value=False)

    @classmethod
    def identity(cls, size: int) -> BitMatrix:
        return cls.generate(width=size, height=size, value=lambda x, y: x == y)

    def __eq__(self, other: Union[List[List[Any]], BitMatrix, bool]) -> bool:
        if isinstance(other, bool):
            return all([all([elem is other for elem in row]) for row in self._values])
        return super().__eq__(other)

    def __and__(self, other: Matrix[bool]) -> Matrix:
        return self.__base_binary_operation_creating_new_entity__(other, lambda x, y: x & y)

    def __or__(self, other: Matrix[bool]) -> Matrix:
        return self.__base_binary_operation_creating_new_entity__(other, lambda x, y: x | y)

    def __xor__(self, other: Matrix[bool]) -> Matrix:
        return self.__base_binary_operation_creating_new_entity__(other, lambda x, y: x ^ y)

    def __iand__(self, other: Matrix[bool]) -> Matrix:
        return self.__base_binary_operation_applying_to_self__(other, lambda x, y: x & y)

    def __ior__(self, other: Matrix[bool]) -> Matrix:
        return self.__base_binary_operation_applying_to_self__(other, lambda x, y: x | y)

    def __ixor__(self, other: Matrix[bool]) -> Matrix:
        return self.__base_binary_operation_applying_to_self__(other, lambda x, y: x ^ y)

    def __neg__(self) -> Self:
        return BitMatrix(
            width=self.width,
            height=self.height,
            values=[[not self._values[j][i] for i in range(self.width)] for j in range(self.height)],
        )

    def __sub__(self, other: Matrix) -> Self:
        return self.__base_binary_operation_creating_new_entity__(other, lambda x, y: x and not y)

    def __isub__(self, other: Matrix) -> Self:
        return self.__base_binary_operation_applying_to_self__(other, lambda x, y: x and not y)
