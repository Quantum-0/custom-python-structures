import unittest

from src.loop_list import LoopList


def lrange(*args, **kwargs):
    return list(range(*args, **kwargs))


class Empty(unittest.TestCase):
    def setUp(self) -> None:
        self.ring = LoopList()

    def test_empty_length(self):
        assert len(self.ring) == 0

    def test_error_deleting(self):
        def deletion():
            del self.ring[0]

        self.assertRaises(ValueError, deletion)

    def test_eq_empty_list(self):
        assert self.ring == []


class Indexing(unittest.TestCase):
    def setUp(self) -> None:
        self.ring = LoopList(range(10))

    def test_length(self):
        assert len(self.ring) == 10

    def test_get_in_range_elements(self):
        assert self.ring[0] == 0
        assert self.ring[5] == 5
        assert self.ring[9] == 9

    def test_get_overflowed_elements(self):
        assert self.ring[10] == 0
        assert self.ring[15] == 5
        assert self.ring[20] == 0
        assert self.ring[29] == 9
        assert self.ring[31] == 1

    def test_get_negative_elements(self):
        assert self.ring[-1] == 9
        assert self.ring[-5] == 5
        assert self.ring[-10] == 0
        assert self.ring[-17] == 3

    def test_set_in_range(self):
        self.ring[5] = 20
        assert self.ring[5] == 20
        assert self.ring[15] == 20
        assert self.ring[-5] == 20
        self.ring[5] = 5
        self.ring[0] = 100
        self.ring[9] = -100
        assert self.ring[0] == self.ring[10] == self.ring[-10] == 100
        assert self.ring[-1] == self.ring[9] == self.ring[19] == -100
        self.ring[0] = 0
        self.ring[9] = 9

    def test_set_negative_and_overflow(self):
        self.ring[-15] = -15
        self.ring[26] = 26
        assert self.ring[-5] == self.ring[5] == self.ring[15] == -15
        assert self.ring[-4] == self.ring[6] == self.ring[16] == 26
        self.ring[-15] = 5
        self.ring[26] = 6

    def tearDown(self) -> None:
        for i in range(10):
            assert self.ring[i] == i
        assert self.ring == lrange(10)


class Slicing(unittest.TestCase):
    def setUp(self) -> None:
        self.ring = LoopList(range(10))

    def test_get_in_range(self):
        assert self.ring[:] == lrange(10)
        assert self.ring[:3] == [0, 1, 2]
        assert self.ring[7:] == [7, 8, 9]
        assert self.ring[4:7] == [4, 5, 6]

    def test_overflow(self):
        assert self.ring[8:13] == [8, 9, 0, 1, 2]
        assert self.ring[22:25] == [2, 3, 4]

    def test_negative(self):
        assert self.ring[-2:3] == [8, 9, 0, 1, 2]
        assert self.ring[-15:-12] == [5, 6, 7]

    def test_overflow_several_times(self):
        assert self.ring[-1:11] == [9] + lrange(10) + [0]
        assert self.ring[-15:25] == lrange(5, 10) + (lrange(10) * 3) + lrange(5)

    def test_slicing_error(self):
        self.assertRaises(IndexError, lambda: self.ring[5:4])
        self.assertRaises(IndexError, lambda: self.ring[0:"test"])
        self.assertRaises(IndexError, lambda: self.ring[{}:[]])


class RotationReverseAndAppend(unittest.TestCase):
    def setUp(self) -> None:
        self.ring = LoopList(range(10))

    def test_reverse(self):
        ring2 = reversed(self.ring)
        ring3 = reversed(ring2)
        assert self.ring == ring3
        assert self.ring != ring2
        l = ring2[:][::-1]
        assert self.ring == l
        self.ring.reverse()
        assert ring2 == self.ring
        self.ring.reverse()

    def test_rotate(self):
        assert self.ring.rotate(len(self.ring)) == lrange(10)
        assert self.ring.rotate(5) == [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
        assert self.ring.rotate(5) == lrange(10)
        assert self.ring.rotate(2) == [2, 3, 4, 5, 6, 7, 8, 9, 0, 1]
        assert self.ring.rotate(8) == lrange(10)
        assert self.ring.rotate(-1) == [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        assert self.ring.rotate(48) == [7, 8, 9, 0, 1, 2, 3, 4, 5, 6]
        assert self.ring.rotate(3) == lrange(10)

    @staticmethod
    def test_append():
        ring = LoopList([1] * 5)
        for i in range(5):
            ring.append(0)
            assert len(ring) == 6 + i
            ring.rotate(1)
        assert ring == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]


class Deletion(unittest.TestCase):
    def setUp(self) -> None:
        self.ring = LoopList(range(10))

    def test_delete(self):
        del self.ring[9]
        assert len(self.ring) == 9
        assert self.ring == lrange(9)
        del self.ring[0]
        assert len(self.ring) == 8
        assert self.ring == lrange(1, 9)
        del self.ring[4]
        assert self.ring == [1, 2, 3, 4, 6, 7, 8]


class JosephusProblem(unittest.TestCase):
    cases = [[2, 1, 2], [5, 2, 3], [7, 5, 6]]

    @staticmethod
    def solve_josephus_problem(n, k):
        r = LoopList(range(1, n + 1))
        while len(r) > 1:
            r.rotate(k - 1)
            del r[0]
        return r[0]

    def test_all_cases(self):
        for case in self.cases:
            n, k, r = case
            assert self.solve_josephus_problem(n, k) == r


class Insert(unittest.TestCase):
    @staticmethod
    def test_insert_in_range():
        for i in range(3):
            l = [1, 2, 3]
            r = LoopList(l)
            r.insert(i, 4)
            l.insert(i, 4)
            assert l == r, (i, l, r)

    @staticmethod
    def test_bound():
        l = [1, 2, 3]
        r1 = LoopList(l)
        r2 = LoopList(l)
        r1.insert(3, 4)
        r2.insert(0, 4)
        assert r1 != r2

    @staticmethod
    def test_bounds_out_of_range():
        l = [1, 2, 3]
        r1 = LoopList(l)
        r2 = LoopList(l)
        r3 = LoopList(l)
        r4 = LoopList(l)
        r1.insert(-3, 4)
        r2.insert(0, 4)
        r3.insert(3, 4)
        r4.insert(6, 4)
        assert r1 == r2
        assert r3 == r4

    @staticmethod
    def test_out_of_base_range():
        l = [1, 2, 3]
        r = LoopList(l)
        r.insert(10, 4)
        assert r == [1, 4, 2, 3]
        r.insert(-2, 5)
        assert r == [1, 4, 5, 2, 3]
        r.insert(-5, 0)
        assert r == [0, 1, 4, 5, 2, 3]


class StringRepresentation(unittest.TestCase):
    @staticmethod
    def test_repr():
        r = LoopList(range(3))
        assert str(r) == "<LoopList[0, 1, 2]>"


if __name__ == "__main__":
    unittest.main()
