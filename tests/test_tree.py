import unittest

from src.tree import Tree, TreeNode


class Empty(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = Tree(None)

    def test_empty_length(self):
        assert len(self.tree) == 1


class AddingDeleting(unittest.TestCase):
    def test_add(self):
        self.tree = Tree(None)
        node = TreeNode(None)
        self.tree += node
        assert len(self.tree) == 2
        assert node in self.tree._children

    def test_delete(self):
        self.tree = Tree(None)
        node = TreeNode(None)
        node_2 = TreeNode(None)
        self.tree += node
        node += node_2
        assert len(self.tree) == 3
        node -= node_2
        assert len(self.tree) == 2
        self.tree -= node
        assert len(self.tree) == 1


class Relationships(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = Tree(0)
        self.node1 = TreeNode(1)
        self.node2 = TreeNode(2)
        self.node3 = TreeNode(3)
        self.node4 = TreeNode(4)
        self.tree += self.node1
        self.tree += self.node2
        self.node2 += self.node3
        self.node3 += self.node4
        print(self.tree.visualize())

    def test_parents(self):
        assert self.node3.is_parent_for(self.node4, on_all_tree=False)
        assert not self.node1.is_parent_for(self.node2, on_all_tree=False)
        assert not self.tree.is_parent_for(self.node4, on_all_tree=False)
        assert self.tree.is_parent_for(self.node4, on_all_tree=True)

    def test_children(self):
        assert self.node4.is_child_for(self.node3, on_all_tree=False)
        assert not self.node2.is_child_for(self.node1, on_all_tree=False)
        assert not self.node4.is_child_for(self.tree, on_all_tree=False)
        assert self.node4.is_child_for(self.tree, on_all_tree=True)

    def test_contains(self):
        for node in [self.tree, self.node1, self.node2, self.node3, self.node4]:
            assert node in self.tree
            assert node.value in self.tree
