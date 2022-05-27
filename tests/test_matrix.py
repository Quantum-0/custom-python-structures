import unittest

from src.matrix import Matrix, NotSquareMatrix


class Empty(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = Matrix(0, 0, [])

    def test_zero_size(self):
        assert self.matrix.width == 0
        assert self.matrix.height == 0
        assert self.matrix.size == (0, 0)


class Equal(unittest.TestCase):
    def setUp(self) -> None:
        self.zero3 = Matrix.zero_matrix(3)
        self.one3 = Matrix.identity(3)
        self.zero2 = Matrix.zero_matrix(2)
        self.one2 = Matrix.identity(2)

    def test_eq(self):
        assert self.zero3 != self.one3
        assert self.zero3 == Matrix.zero_matrix(3)
        assert self.zero3 == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        assert self.one2 == [[1, 0], [0, 1]]
        assert self.zero3 != self.zero2
        assert self.one3 != self.one2

    def test_contains(self):
        zero2x3 = [[0, 0], [0, 0], [0, 0]]
        one2x3 = [[1, 0], [0, 1], [0, 0]]
        assert self.zero2 in self.zero3
        assert self.one2 in self.one3
        assert zero2x3 in self.zero3
        assert self.zero2 in Matrix.from_nested_list(zero2x3)
        assert one2x3 in self.one3
        assert self.one2 in Matrix.from_nested_list(one2x3)


class PreDefined(unittest.TestCase):
    def test_identity_matrix(self):
        assert Matrix.identity(1) == [[1]]
        assert Matrix.identity(2) == [[1, 0], [0, 1]]
        assert Matrix.identity(3) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert Matrix.identity(4) == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    def test_zero_matrix(self):
        assert Matrix.zero_matrix(1) == [[0]]
        assert Matrix.zero_matrix(2) == [[0, 0], [0, 0]]
        assert Matrix.zero_matrix(3) == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        assert Matrix.zero_matrix(4) == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


class Generation(unittest.TestCase):
    def test_identity(self):
        m1 = Matrix(3, 3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        m2 = Matrix.from_lists([1, 0, 0], [0, 1, 0], [0, 0, 1])
        m3 = Matrix.from_nested_list([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        m4 = Matrix.identity(3)
        m5 = Matrix.generate(3, 3, lambda x, y: 1 if x == y else 0)
        m6 = Matrix.from_joined_lists(3, values=[1, 0, 0, 0, 1, 0, 0, 0, 1])
        m7 = Matrix.from_joined_lists(3, 3, values=[1, 0, 0, 0, 1, 0, 0, 0, 1])
        assert m1._values == m2._values == m3._values == m4._values == m5._values == m6._values == m7._values

    def test_zero(self):
        m1 = Matrix.generate(3, 3, 0)
        m2 = Matrix.zero_matrix(3)
        m3 = Matrix.generate(3, 3, lambda x, y: 0)
        m4 = Matrix.generate(3, 3, lambda: 0)
        m5 = Matrix.from_lists([0, 0, 0], [0, 0, 0], [0, 0, 0])
        m6 = Matrix.from_nested_list([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        m7 = Matrix(3, 3, [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        m8 = Matrix.from_joined_lists(3, values=[0] * 9)
        m9 = Matrix.from_joined_lists(3, 3, values=[0] * 9)
        m10 = Matrix.generate(3, 3, 0)
        assert m1._values == m2._values == m3._values == m4._values == m5._values == m6._values == m7._values == m8._values == m9._values == m10._values

    def test_generation_lambda(self):
        m = Matrix.generate(3, 3, range(9).__iter__())
        assert m == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        m = Matrix.generate(3, 3, lambda x, y: x + y)
        assert m == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        m = Matrix.from_joined_lists(3, values=range(9))
        assert m == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


class Indexing(unittest.TestCase):
    def setUp(self) -> None:
        self.m = Matrix.from_joined_lists(3, values=range(6))

    def test_indexing(self):
        assert self.m[0, 0] == 0
        assert self.m[1, 0] == 1
        assert self.m[2, 0] == 2
        assert self.m[0, 1] == 3
        assert self.m[1, 1] == 4
        assert self.m[2, 1] == 5

    def test_index_error(self):
        self.assertRaises(IndexError, lambda: self.m[-1, 0])
        self.assertRaises(IndexError, lambda: self.m[0, -1])
        self.assertRaises(IndexError, lambda: self.m[3, 0])
        self.assertRaises(IndexError, lambda: self.m[0, 2])
        self.assertRaises(IndexError, lambda: self.m[3, 2])
        self.assertRaises(IndexError, lambda: self.m[-1, -1])

        self.assertRaises(IndexError, lambda: self.m[0])  # noqa
        self.assertRaises(IndexError, lambda: self.m[0, 1, 2])  # noqa
        self.assertRaises(IndexError, lambda: self.m[1.5, 2])  # noqa
        self.assertRaises(IndexError, lambda: self.m[0:2])  # noqa


class Slicing(unittest.TestCase):
    def setUp(self) -> None:
        self.m = Matrix.from_joined_lists(4, values=range(12))

    def test_horizontal_slice(self):
        assert self.m[1, :] == [4, 5, 6, 7]
        assert self.m[0, 1:3] == [1, 2]
        assert self.m[2, 3:] == [11, 12]

    def test_vertical_slice(self):
        pass

    def test_both_slice(self):
        pass

    def test_index_error(self):
        pass


class Math(unittest.TestCase):
    def test_mul_to_number(self):
        pass

    def test_add_matrix(self):
        pass

    def test_sub_matrix(self):
        pass

    def test_mul_matrix(self):
        pass

    def test_inverse_matrix(self):
        # A * ~A == 1
        pass

    def test_trace(self):
        pass

    def test_determinant(self):
        pass
