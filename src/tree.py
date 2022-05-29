from __future__ import annotations

from collections.abc import Sized
from typing import Any, Set, Union, Optional


class TreeNode(Sized):
    def __init__(self, value: Any):
        self.value: Any = value
        self._children: Set[TreeNode] = set()
        self._parent: Optional[TreeNode] = None

    def __del__(self):
        for child in self._children:
            child.__del__()

    def __iadd__(self, other: TreeNode) -> TreeNode:
        self._children.add(other)
        other._parent = self
        return self

    def __isub__(self, other: TreeNode) -> TreeNode:
        self._children.remove(other)
        return self

    def __len__(self):
        return sum(len(node) for node in self._children) + 1

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.value})>"

    def __contains__(self, item: Union[TreeNode, Any]):
        if item == self or item == self.value:
            return True
        return any([item in node for node in self._children])

    def visualize(self):
        return self.__repr_indent__()[::-1].replace("┣", "┗", 1)[::-1]

    def __repr_indent__(self, indent=0):
        str_indent = " ┣" + "━" * (indent * 3 - 2) if indent > 0 else ""
        return (
            str_indent
            + str(self)
            + "".join(["\n" + node.__repr_indent__(indent=indent + 1) for node in self._children])
        )

    def is_child_for(self, parent: TreeNode, on_all_tree: bool) -> bool:
        if self in parent._children:
            return True
        if not on_all_tree:
            return False
        return any([self.is_child_for(parent=node, on_all_tree=True) for node in parent._children])

    def is_parent_for(self, child: TreeNode, on_all_tree: bool) -> bool:
        if child in self._children:
            return True
        if not on_all_tree:
            return False
        return any([node.is_parent_for(child=child, on_all_tree=True) for node in self._children])


class TreeRoot(TreeNode):
    def __init__(self, root_value: Any):
        super().__init__(root_value)

    # TODO: get list of nodes


Tree = TreeRoot
