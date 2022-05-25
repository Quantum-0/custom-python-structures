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
    tree = None
    node1 = None
    node2 = None
    node3 = None
    node4 = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = Tree(0)
        cls.node1 = TreeNode(1)
        cls.node2 = TreeNode(2)
        cls.node3 = TreeNode(3)
        cls.node4 = TreeNode(4)
        cls.tree += cls.node1
        cls.tree += cls.node2
        cls.node2 += cls.node3
        cls.node3 += cls.node4

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


class Representation(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = Tree('Root')
        self.child = TreeNode('Child')
        self.tree += self.child

    def test_repr(self):
        assert str(self.tree) == "<TreeRoot(Root)>"
        assert str(self.child) == "<TreeNode(Child)>"

    def test_visualization(self):
        assert self.tree.visualize() == '<TreeRoot(Root)>\n ┗━<TreeNode(Child)>'
        assert self.child.visualize() == str(self.child)
