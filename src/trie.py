from collections.abc import Mapping
from typing import Any, Iterator

from src.tree import TreeNode


class TrieNode(TreeNode, Mapping):
    """Prefix Tree implimentation"""

    def __iter__(self) -> Iterator:
        raise NotImplementedError()

    def __getitem__(self, key: str) -> Any:
        for node in self._children:
            if key.startswith(node.key):
                return node[key]
        raise IndexError()

    def __setitem__(self, key: str, value: Any):
        if not self.is_root:
            raise KeyError("Cannot get element from Trie node")

    def __iadd__(self, other: TreeNode) -> TreeNode:
        self._children.add(other)
        other._parent = self
        other._root = self.root
        return self

    def __isub__(self, other: TreeNode) -> TreeNode:
        self._children.remove(other)
        other._root = None
        return self

    @property
    def key(self):
        return self.parent.key + self.key

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.key} = {self.value})>"


class TrieRoot(TrieNode):
    def __init__(self):
        super().__init__("")
        self.root = self


Trie = TrieRoot
