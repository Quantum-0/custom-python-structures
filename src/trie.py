from src.tree import TreeNode


class TrieNode(TreeNode):
    """Prefix Tree implimentation"""

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
    def value(self):
        return self.parent.value + self.value

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.value})>"


class TrieRoot(TrieNode):
    def __init__(self):
        super().__init__('')
        self.root = self

Trie = TrieRoot