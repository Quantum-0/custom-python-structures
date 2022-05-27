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
        # 0 1 2
        # 3 4 5

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
        self.m = Matrix.from_joined_lists(4, values=range(20))
        # 0 1 2 3
        # 4 5 6 7
        # 8 9 10 11
        # 12 13 14 15
        # 16 17 18 19

    def test_vertical_slice(self):
        assert self.m[1, :] == [1, 5, 9, 13, 17], self.m[1, :]
        assert self.m[1, :] == [self.m[1, 0], self.m[1, 1], self.m[1, 2], self.m[1, 3], self.m[1, 4]]
        assert self.m[0, 1:3] == [4, 8], self.m[0, 1:3]
        assert self.m[2, 2:] == [10, 14, 18]

    def test_horizontal_slice(self):
        assert self.m[:, 1] == [4, 5, 6, 7], self.m[:, 1]
        assert self.m[1:3, 0] == [1, 2], self.m[1:3, 0]
        assert self.m[2:, 2] == [10, 11]

    def test_both_slice(self):
        assert self.m[:, :] == self.m._values
        assert self.m[0:3, 0:3] == [[0, 1, 2], [4, 5, 6], [8, 9, 10]]
        assert self.m[1:3, 1:4] == [[5, 6], [9, 10], [13, 14]]
        assert self.m[2:, 2:] == [[10, 11], [14, 15], [18, 19]]

    def test_index_error(self):
        pass


class Math(unittest.TestCase):
    def setUp(self) -> None:
        self.A = Matrix.from_joined_lists(4, values=range(20))
        self.B = Matrix.from_lists([1, 2], [3, 4], [5, 6])
        self.C = Matrix.from_lists([1, 3, 5], [2, 4, 6])
        self.D = Matrix.from_lists([1, 2], [3, 4])
        self.E = Matrix.identity(3)
        self.F = Matrix.from_lists([1, 1], [2, 2])
        self.G = Matrix.from_lists([-1, -2], [1, 2])
        self.H = Matrix.from_lists([0, -1], [3, 4])

    def test_mul_to_number(self):
        assert self.E * 3 == [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
        assert self.E == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.E *= 3
        assert self.E == [[3, 0, 0], [0, 3, 0], [0, 0, 3]], self.E
        assert self.E * (1 / 3) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert self.E == [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
        self.E /= 3
        assert self.E == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        pass

    def test_add_matrix(self):
        assert self.G + self.F == self.H
        assert self.F + self.G == self.H
        assert self.E + Matrix.zero_matrix(self.E.width) == self.E

        self.G += self.F
        assert self.G == self.H
        pass

    def test_sub_matrix(self):
        assert self.D - self.D == Matrix.zero_matrix(self.D.width)

        self.E -= self.E
        assert self.E == Matrix.zero_matrix(self.E.width)
        pass

    def test_mul_matrix(self):
        assert Matrix(1, 2, [[1,2]]) * Matrix(2, 1, [[1], [2]]) == [[5]]
        assert Matrix.from_nested_list([[1,2,3],[3,4,2],[3,2,1]]) * Matrix.from_nested_list([[1,1,1],[3,4,2],[3,2,1]])\
               == Matrix.from_nested_list([[16,15,8],[21,23,13],[12,13,8]])
        assert Matrix.from_lists([2,0],[1,9]) * Matrix.from_lists([3,9],[4,7]) == Matrix.from_lists([6,18],[39,72])

    def test_inverse_matrix(self):
        # A * ~A == 1
        assert ~self.D == [[-2, 1], [1.5, -0.5]]
        #assert self.D * ~self.D == Matrix.identity(self.D.width)
        #assert self.A * ~self.A == Matrix.identity(self.A.width)
        #assert ~self.A * self.A == Matrix.identity(self.A.height)

    def test_trace(self):
        assert self.D.trace == 5

    def test_determinant(self):
        assert self.D.determinant == -2

    def test_transpose(self):
        self.B.transpose()
        assert self.B == self.C


class MirroringAndRotating(unittest.TestCase):
    def setUp(self) -> None:
        self.M = Matrix.from_joined_lists(3, values=range(9))
        # 0 1 2
        # 3 4 5
        # 6 7 8

    def test_vertical_mirroring(self):
        assert self.M.mirrored_verticaly == [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    def test_horizontal_mirroring(self):
        assert self.M.mirrored_horizontaly == [[2, 1, 0], [5, 4, 3], [8, 7, 6]]

    def test_other_mirroring(self):
        assert self.M.mirrored_horizontaly.mirrored_horizontaly == self.M
        assert self.M.mirrored_verticaly.mirrored_verticaly == self.M
        assert self.M.mirrored_horizontaly.mirrored_verticaly == [[8, 7, 6], [5, 4, 3], [2, 1, 0]]

    def test_clockwise_rotation(self):
        assert self.M.rotated_clockwise == [[6, 3, 0], [7, 4, 1], [8, 5, 2]], self.M.rotated_clockwise
        assert self.M.rotated_clockwise.rotated_clockwise.rotated_clockwise.rotated_clockwise == self.M

    def test_counterclockwise_rotation(self):
        assert self.M.rotated_counterclockwise == [[2, 5, 8], [1, 4, 7], [0, 3, 6]]
        assert self.M.rotated_counterclockwise.rotated_counterclockwise.rotated_counterclockwise.rotated_counterclockwise == self.M

    def test_other_rotation(self):
        assert self.M.rotated_clockwise.rotated_counterclockwise == self.M

    def test_other(self):
        assert self.M.rotated_clockwise.rotated_clockwise == self.M.mirrored_verticaly.mirrored_horizontaly


class LogicWithBitMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self.A = Matrix(2, 2, [[True, True], [False, False]])
        self.B = Matrix(2, 2, [[True, False], [True, False]])

    def test_compare_to_bool(self):
        # Matrix.identity(3) == True == error
        assert self.A != True
        assert self.B != True
        assert self.A != False
        assert self.B != False
        assert Matrix.from_nested_list([[True, True], [True, True]]) == True
        assert Matrix.from_nested_list([[False, False], [False, False]]) == False
        assert Matrix.from_nested_list([[True, True], [True, True]]) != False
        assert Matrix.from_nested_list([[False, False], [False, False]]) != True

    def test_and(self):
        assert self.A & self.A == self.A
        assert self.A & self.B == self.B & self.A
        assert self.A & self.B == [[True, False], [False, False]]
        assert self.A & True == self.A
        assert self.A & False == False

    def test_or(self):
        assert self.A | self.A == self.A
        assert self.A | self.B == self.B | self.A
        assert self.A | self.B == [[True, True], [True, False]]
        assert self.A | True == True
        assert self.A | False == self.A

    def test_not(self):
        assert -self.A == [[False, False], [True, True]]
        assert -self.B == [[False, True], [False, True]]

    def test_xor(self):
        assert self.A ^ self.B == self.B ^ self.A == [[False, True], [True, False]]
        assert self.A | self.B - self.A & self.B == self.A ^ self.B
