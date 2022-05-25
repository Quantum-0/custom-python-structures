from collections.abc import Sequence
from typing import Union, Any, Optional, List


class LoopList(Sequence):
    def __init__(self, values: Optional[Union[List, Sequence, range]] = None):
        self._internal_list = list(values) if values else []

    def insert(self, index: int, value: Any) -> None:
        if index > 0:
            self._internal_list.insert((index - 1) % len(self) + 1, value)
        else:
            self._internal_list.insert(index % len(self), value)

    def __getitem__(self, i: Union[int, slice]) -> Any:
        if isinstance(i, int):
            return self._internal_list[i % len(self)]
        if not isinstance(i, slice):
            raise IndexError('Index must be int or slice')

        if not isinstance(i.start, int) and i.start is not None:
            raise IndexError()
        if not isinstance(i.stop, int) and i.stop is not None:
            raise IndexError()
        begin = i.start or 0
        end = i.stop or len(self)
        if begin > end:
            raise IndexError()
        begin_mod = begin % len(self)
        end_mod = end % len(self)
        full_cycles = (end // len(self)) - (begin // len(self))
        if full_cycles:
            return (
                self._internal_list[begin_mod:]
                + self._internal_list * (full_cycles - 1)
                + self._internal_list[:end_mod]
            )
        return self._internal_list[begin_mod:end_mod]

    def __setitem__(self, i: int, value: Any) -> None:
        if isinstance(i, int):
            self._internal_list[i % len(self)] = value
        else:
            raise IndexError('Index must be int')

    def __delitem__(self, i: int) -> None:
        if len(self) == 0:
            raise ValueError(self.__class__.__name__ + " is Empty")
        if not isinstance(i, int):
            raise IndexError("Index must be int")
        del self._internal_list[i % len(self)]

    def __len__(self) -> int:
        return len(self._internal_list)

    def append(self, value: Any) -> None:
        self._internal_list.append(value)

    def __eq__(self, other) -> bool:
        if isinstance(other, list):
            return self._internal_list == other
        if isinstance(other, LoopList):
            return self._internal_list == other._internal_list
        raise TypeError("Cannot compare")

    def rotate(self, indexes: int):
        i = indexes % len(self)
        self._internal_list = self._internal_list[i:] + self._internal_list[:i]
        return self

    def __iter__(self):
        yield from self._internal_list

    def __reversed__(self):
        return LoopList(self._internal_list[::-1])

    def reverse(self):
        self._internal_list = self._internal_list[::-1]

    def __repr__(self):
        return f"<LoopList{self._internal_list}>"
