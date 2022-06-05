import unittest

from src.matrix.bit_matrix import BitMatrix
from src.matrix.matrix import NotSquareMatrix, Matrix
from src.matrix.matrix_iterator import MatrixIterator
from src.matrix.numeric_matrix import NumericMatrix


class Empty(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = Matrix(0, 0, [])

    def test_zero_size(self):
        assert self.matrix.width == 0
        assert self.matrix.height == 0
        assert self.matrix.size == (0, 0)


class Representation(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix1 = Matrix(2, 2, [[1, 2], [3, 4]])
        self.matrix2 = NumericMatrix(2, 2, [[1, 2], [3, 4]])
        self.matrix3 = BitMatrix(2, 2, [[True, False], [False, True]])

    def test_repr(self):
        assert str(self.matrix1) == "<Matrix([[1, 2], [3, 4]])>"
        assert str(self.matrix2) == "<NumericMatrix([[1, 2], [3, 4]])>"
        assert str(self.matrix3) == "<BitMatrix([[True, False], [False, True]])>"


class Equal(unittest.TestCase):
    def setUp(self) -> None:
        self.zero3 = NumericMatrix.zero_matrix(3)
        self.one3 = NumericMatrix.identity(3)
        self.zero2 = NumericMatrix.zero_matrix(2)
        self.one2 = NumericMatrix.identity(2)
        self.m3 = Matrix.from_joined_lists(3, values=range(9))
        self.m2 = Matrix.from_joined_lists(2, values=range(4))

    def test_eq(self):
        assert self.zero3 != self.one3
        assert self.zero3 == NumericMatrix.zero_matrix(3)
        assert self.zero3 == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        assert self.one2 == [[1, 0], [0, 1]]
        assert self.zero3 != self.zero2
        assert self.one3 != self.one2
        assert self.zero3.main_diagonal == [0, 0, 0]
        assert self.one2.main_diagonal == [1, 1]
        assert self.m2 != 42

    def test_contains(self):
        zero2x3 = [[0, 0], [0, 0], [0, 0]]
        one2x3 = [[1, 0], [0, 1], [0, 0]]
        one3x3_s = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert self.zero2 in self.zero3
        assert self.one2 in self.one3
        assert self.one2 in NumericMatrix.from_nested_list(one3x3_s)
        assert zero2x3 in self.zero3
        assert self.zero2 in NumericMatrix.from_nested_list(zero2x3)
        assert one2x3 in self.one3
        assert self.one2 in NumericMatrix.from_nested_list(one2x3)
        assert self.zero3 not in self.zero2
        assert self.m3 not in self.m2
        assert self.m2 not in self.m3


class PreDefined(unittest.TestCase):
    def test_identity_matrix(self):
        assert NumericMatrix.identity(1) == [[1]]
        assert NumericMatrix.identity(2) == [[1, 0], [0, 1]]
        assert NumericMatrix.identity(3) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert NumericMatrix.identity(4) == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        assert NumericMatrix.identity(3).is_identity
        assert not NumericMatrix.identity(3).is_zero

    def test_zero_matrix(self):
        assert NumericMatrix.zero_matrix(1) == [[0]]
        assert NumericMatrix.zero_matrix(2) == [[0, 0], [0, 0]]
        assert NumericMatrix.zero_matrix(3) == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        assert NumericMatrix.zero_matrix(4) == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        assert not NumericMatrix.zero_matrix(3).is_identity
        assert NumericMatrix.zero_matrix(3).is_zero


class Generation(unittest.TestCase):
    def test_identity(self):
        m1 = NumericMatrix(3, 3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        m2 = NumericMatrix.from_lists([1, 0, 0], [0, 1, 0], [0, 0, 1])
        m3 = NumericMatrix.from_nested_list([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        m4 = NumericMatrix.identity(3)
        m5 = NumericMatrix.generate(3, 3, lambda x, y: 1 if x == y else 0)
        m6 = NumericMatrix.from_joined_lists(3, values=[1, 0, 0, 0, 1, 0, 0, 0, 1])
        m7 = NumericMatrix.from_joined_lists(3, 3, values=[1, 0, 0, 0, 1, 0, 0, 0, 1])
        assert m1._values == m2._values == m3._values == m4._values == m5._values == m6._values == m7._values

    def test_zero(self):
        m1 = NumericMatrix.generate(3, 3, 0)
        m2 = NumericMatrix.zero_matrix(3)
        m3 = NumericMatrix.generate(3, 3, lambda x, y: 0)
        m4 = NumericMatrix.generate(3, 3, lambda: 0)
        m5 = NumericMatrix.from_lists([0, 0, 0], [0, 0, 0], [0, 0, 0])
        m6 = NumericMatrix.from_nested_list([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        m7 = NumericMatrix(3, 3, [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        m8 = NumericMatrix.from_joined_lists(3, values=[0] * 9)
        m9 = NumericMatrix.from_joined_lists(3, 3, values=[0] * 9)
        m10 = NumericMatrix.generate(3, 3, 0)
        m11 = NumericMatrix.zero_matrix(size=(3, 3))
        assert (
            m1._values
            == m2._values
            == m3._values
            == m4._values
            == m5._values
            == m6._values
            == m7._values
            == m8._values
            == m9._values
            == m10._values
            == m11._values
        )

    def test_generation_lambda(self):
        m = NumericMatrix.generate(3, 3, range(9).__iter__())
        assert m == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        m = NumericMatrix.generate(3, 3, lambda x, y: x + y)
        assert m == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        m = NumericMatrix.from_joined_lists(3, values=range(9))
        assert m == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        m = NumericMatrix.generate(3, 3, [1, 2, 3], by_rows=True)
        assert m == [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        m = NumericMatrix.generate(3, 4, lambda row_id: [row_id, row_id*2, row_id**2], by_rows=True)
        assert m == [[0, 0, 0], [1, 2, 1], [2, 4, 4], [3, 6, 9]]
        m = NumericMatrix.generate(3, 2, lambda: list((1, 2, 3)), by_rows=True)
        assert m == [[1, 2, 3], [1, 2, 3]]

        def range_lists_iterator(count: int, list_len: int):
            rng = range(count)
            for element in rng:
                yield [element] * list_len

        m = NumericMatrix.generate(3, 4, range_lists_iterator(4, 3), by_rows=True)
        assert m == [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]

    def test_errors(self):
        self.assertRaises(ValueError, lambda: NumericMatrix.from_lists())
        self.assertRaises(ValueError, lambda: NumericMatrix.from_lists([]))
        self.assertRaises(ValueError, lambda: NumericMatrix.from_joined_lists(3, values=[1, 2, 3, 4, 5]))
        self.assertRaises(ValueError, lambda: NumericMatrix.from_joined_lists(3, 3, values=[1, 2, 3, 4, 5, 6]))
        self.assertRaises(ValueError, lambda: NumericMatrix.from_nested_list([]))
        self.assertRaises(TypeError, lambda: NumericMatrix.from_nested_list("test"))  # noqa
        self.assertRaises(ValueError, lambda: NumericMatrix.from_nested_list([[], [], []]))
        self.assertRaises(ValueError, lambda: NumericMatrix.from_nested_list([[1, 2], [3, 4], [5, 6, 7], [8, 9]]))
        self.assertRaises(ValueError, lambda: NumericMatrix.generate(3, 3, lambda x, y, z: x + y + z))
        self.assertRaises(ValueError, lambda: NumericMatrix.generate(3, 3, lambda x, y: x + y, by_rows=True))
        self.assertRaises(TypeError, lambda: NumericMatrix.zero_matrix("test"))  # noqa
        self.assertRaises(TypeError, lambda: NumericMatrix.zero_matrix((1, 2, 3)))  # noqa
        self.assertRaises(TypeError, lambda: BitMatrix.zero_matrix("test"))  # noqa
        self.assertRaises(TypeError, lambda: BitMatrix.zero_matrix((1, 2, 3)))  # noqa
        self.assertRaises(ValueError, lambda: NumericMatrix.generate(-1, 2, lambda x, y, z: x + y + z))
        self.assertRaises(ValueError, lambda: NumericMatrix.generate(4, 0, lambda x, y, z: x + y + z))
        self.assertRaises(TypeError, lambda: NumericMatrix.generate("test", 2, lambda x, y, z: x + y + z))


class Indexing(unittest.TestCase):
    def setUp(self) -> None:
        self.m = NumericMatrix.from_joined_lists(3, values=range(6))
        # 0 1 2
        # 3 4 5

    def test_indexing(self):
        assert self.m[0, 0] == 0
        assert self.m[1, 0] == 1
        assert self.m[2, 0] == 2
        assert self.m[0, 1] == 3
        assert self.m[1, 1] == 4
        assert self.m[2, 1] == 5

    def test_setting(self):
        for i in range(3):
            for j in range(2):
                assert self.m[i, j] == i + j * 3
                self.m[i, j] = 25
                assert self.m[i, j] == 25

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

        def set_val(index, value):
            self.m[index] = value

        self.assertRaises(IndexError, lambda: set_val(0, 0))
        self.assertRaises(IndexError, lambda: set_val((), 0))
        self.assertRaises(IndexError, lambda: set_val((1, 2, 3), 0))
        self.assertRaises(IndexError, lambda: set_val((slice(1, 2), "test"), 0))
        self.assertRaises(IndexError, lambda: set_val((slice(1, 2), 5), 0))
        self.assertRaises(IndexError, lambda: set_val((-1, 1), 0))


class Slicing(unittest.TestCase):
    # TODO: Add test setitem

    def setUp(self) -> None:
        self.m = NumericMatrix.from_joined_lists(4, values=range(20))
        # 0 1 2 3
        # 4 5 6 7
        # 8 9 10 11
        # 12 13 14 15
        # 16 17 18 19
        self.m2 = NumericMatrix.from_joined_lists(2, values=range(4))
        self.m3 = NumericMatrix.from_joined_lists(3, values=range(9))

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

    def test_minor(self):
        assert self.m3.get_minor(0, 0) == Matrix(2, 2, [[4, 5], [7, 8]])
        assert self.m3.get_minor(1, 1) == Matrix(2, 2, [[0, 2], [6, 8]])
        assert self.m3.get_minor(2, 2) == self.m3[:2, :2]
        assert self.m3.get_minor(0, 0).get_minor(1, 1) == [[4]]
        self.assertRaises(ValueError, lambda: self.m3.get_minor(-1, 2))
        self.assertRaises(ValueError, lambda: self.m3.get_minor(1, 3))
        self.assertRaises(TypeError, lambda: self.m3.get_minor("test", "test"))  # noqa


class Math(unittest.TestCase):
    def setUp(self) -> None:
        self.A = NumericMatrix.from_joined_lists(4, values=range(20))
        self.B = NumericMatrix.from_lists([1, 2], [3, 4], [5, 6])
        self.C = NumericMatrix.from_lists([1, 3, 5], [2, 4, 6])
        self.D = NumericMatrix.from_lists([1, 2], [3, 4])
        self.E = NumericMatrix.identity(3)
        self.F = NumericMatrix.from_lists([1, 1], [2, 2])
        self.G = NumericMatrix.from_lists([-1, -2], [1, 2])
        self.H = NumericMatrix.from_lists([0, -1], [3, 4])
        self.I = NumericMatrix.from_lists([1, 3, 5], [2, 4, 6], [0, -1, -2])

    def test_mul_to_number(self):
        assert self.E * 3 == [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
        assert self.E == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.E *= 3
        assert self.E == [[3, 0, 0], [0, 3, 0], [0, 0, 3]], self.E
        assert self.E * (1 / 3) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert self.E == [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
        self.E /= 3
        assert self.E == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.assertRaises(AttributeError, lambda: self.C * "str")

        def test1():
            self.C *= "str"

        def test2():
            self.C /= "str"

        self.assertRaises(AttributeError, test1)
        self.assertRaises(AttributeError, test2)

    def test_add_matrix(self):
        assert self.G + self.F == self.H
        assert self.F + self.G == self.H
        assert self.E + NumericMatrix.zero_matrix(self.E.width) == self.E

        self.G += self.F
        assert self.G == self.H
        self.assertRaises(AttributeError, lambda: self.C + self.D)

    def test_sub_matrix(self):
        assert self.D - self.D == NumericMatrix.zero_matrix(self.D.width)

        self.E -= self.E
        assert self.E == NumericMatrix.zero_matrix(self.E.width)
        self.assertRaises(AttributeError, lambda: self.C - self.D)

    def test_neg_matrix(self):
        assert -self.F == [[-1, -1], [-2, -2]]

    def test_mul_matrix(self):
        m = NumericMatrix(1, 2, [[1, 2]])
        m2 = NumericMatrix(2, 1, [[1], [2]])
        assert m * m2 == [[5]]
        assert m == [[1, 2]]
        m *= m2
        assert m == [[5]]
        assert NumericMatrix.from_nested_list([[1, 2, 3], [3, 4, 2], [3, 2, 1]]) * NumericMatrix.from_nested_list(
            [[1, 1, 1], [3, 4, 2], [3, 2, 1]]
        ) == NumericMatrix.from_nested_list([[16, 15, 8], [21, 23, 13], [12, 13, 8]])
        assert NumericMatrix.from_lists([2, 0], [1, 9]) * NumericMatrix.from_lists(
            [3, 9], [4, 7]
        ) == NumericMatrix.from_lists([6, 18], [39, 72])
        assert self.I * self.E == self.I
        assert self.E * self.I == self.I
        self.assertRaises(AttributeError, lambda: self.E * self.F)

    def test_inverse_matrix(self):
        # A * ~A == 1
        assert ~self.D == [[-2, 1], [1.5, -0.5]]
        self.assertRaises(NotImplementedError, lambda: ~self.A)
        # assert self.D * ~self.D == Matrix.identity(self.D.width)
        # assert self.A * ~self.A == Matrix.identity(self.A.width)
        # assert ~self.A * self.A == Matrix.identity(self.A.height)

    def test_trace(self):
        assert self.D.trace == 5
        self.assertRaises(NotSquareMatrix, lambda: self.C.trace)

    def test_determinant(self):
        assert self.D.determinant == -2
        # TODO: n=3,4,5

    def test_transpose(self):
        self.B.transpose()
        assert self.B == self.C


class MirroringAndRotating(unittest.TestCase):
    def setUp(self) -> None:
        self.M = NumericMatrix.from_joined_lists(3, values=range(9))
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
        assert (
            self.M.rotated_counterclockwise.rotated_counterclockwise.rotated_counterclockwise.rotated_counterclockwise
            == self.M
        )

    def test_other_rotation(self):
        assert self.M.rotated_clockwise.rotated_counterclockwise == self.M

    def test_other(self):
        assert self.M.rotated_clockwise.rotated_clockwise == self.M.mirrored_verticaly.mirrored_horizontaly


class LogicWithBitMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self.A = BitMatrix(2, 2, [[True, True], [False, False]])
        self.B = BitMatrix(2, 2, [[True, False], [True, False]])
        self.C = BitMatrix(1, 1, [[True]])
        self.D = BitMatrix.zero_matrix(3)
        # self.D1 = Matrix.zero_matrix(3)
        self.E = BitMatrix.identity(3)
        # self.E1 = Matrix.identity(3)
        self.F = BitMatrix.zero_matrix(size=(3, 2))

    def test_boolean_generator(self):
        assert self.D == False
        assert all(self.E.main_diagonal)
        assert self.E == [[True, False, False], [False, True, False], [False, False, True]]
        assert self.F == [[False, False, False], [False, False, False]]

    def test_compare_to_bool(self):
        assert BitMatrix.identity(3) != False
        assert self.A != True
        assert self.B != True
        assert self.A != False
        assert self.B != False
        assert BitMatrix.from_nested_list([[True, True], [True, True]]) == True
        assert BitMatrix.from_nested_list([[False, False], [False, False]]) == False
        assert BitMatrix.from_nested_list([[True, True], [True, True]]) != False
        assert BitMatrix.from_nested_list([[False, False], [False, False]]) != True
        assert self.C == True
        assert self.C != False

    def test_and(self):
        assert self.A & self.A == self.A
        assert self.A & self.B == self.B & self.A
        assert self.A & self.B == [[True, False], [False, False]]
        self.assertRaises(AttributeError, lambda: self.A & self.C)
        self.assertRaises(AttributeError, lambda: self.A & 123)  # noqa
        assert self.A == [[True, True], [False, False]]
        self.A &= self.B
        assert self.A == [[True, False], [False, False]]

    def test_or(self):
        assert self.A | self.A == self.A
        assert self.A | self.B == self.B | self.A
        assert self.A | self.B == [[True, True], [True, False]]
        self.assertRaises(AttributeError, lambda: self.A | self.C)
        self.assertRaises(AttributeError, lambda: self.A | 123)  # noqa
        assert self.A == [[True, True], [False, False]]
        self.A |= self.B
        assert self.A == [[True, True], [True, False]]

    def test_not(self):
        assert not self.A == [[False, False], [True, True]]
        assert not self.B == [[False, True], [False, True]]
        assert -self.A == [[False, False], [True, True]]
        assert -self.B == [[False, True], [False, True]]

    def test_minus(self):
        assert self.A - self.B == [[False, True], [False, False]]
        assert self.B - self.A == [[False, False], [True, False]]
        self.assertRaises(AttributeError, lambda: self.A - self.C)
        self.assertRaises(AttributeError, lambda: self.A - 123)  # noqa
        assert self.A == [[True, True], [False, False]]
        self.A -= self.B
        assert self.A == [[False, True], [False, False]], self.A

    def test_xor(self):
        assert self.A ^ self.B == self.B ^ self.A == [[False, True], [True, False]]
        assert (self.A | self.B) - (self.A & self.B) == self.A ^ self.B
        self.assertRaises(AttributeError, lambda: self.A ^ self.C)
        self.assertRaises(AttributeError, lambda: self.A ^ 123)  # noqa
        assert self.A == [[True, True], [False, False]]
        self.A ^= self.B
        assert self.A == [[False, True], [True, False]]

    def test_errors(self):
        def test1():
            self.A &= self.C

        def test2():
            self.A |= self.C

        def test3():
            self.A ^= self.C

        def test4():
            self.A &= "test"

        self.assertRaises(AttributeError, test1)
        self.assertRaises(AttributeError, test2)
        self.assertRaises(AttributeError, test3)
        self.assertRaises(AttributeError, test4)


class Iterators(unittest.TestCase):
    def setUp(self) -> None:
        self.m = NumericMatrix.from_joined_lists(3, values=range(9))

    def test_default_iterator(self):
        for index, item in enumerate(MatrixIterator.iterate(self.m, Matrix.Walkthrow.DEFAULT)):
            assert index == item

    def test_reversed_iterator(self):
        for index, item in enumerate(MatrixIterator.iterate(self.m, Matrix.Walkthrow.REVERSED)):
            assert index == 8 - item

    def test_snake_iterator(self):
        assert list(MatrixIterator.iterate(self.m, Matrix.Walkthrow.SNAKE)) == [0, 1, 2, 5, 4, 3, 6, 7, 8]

    @unittest.skip("Not Implemented")
    def test_spiral_iterator(self):
        assert list(MatrixIterator.iterate(self.m, Matrix.Walkthrow.SPIRAL)) == [0, 1, 2, 5, 8, 7, 6, 3, 4]

    def test_rows_iterator(self):
        assert list(MatrixIterator.iterate(self.m, Matrix.Walkthrow.ROWS)) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    def test_columns_iterator(self):
        assert list(MatrixIterator.iterate(self.m, Matrix.Walkthrow.COLUMNS)) == [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
