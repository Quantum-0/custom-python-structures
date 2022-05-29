# Custom-Python-Structures

Practice custom data structures.

*Also testing Github Actions on that repo.*

# Badges

[![Tests](https://github.com/Quantum-0/custom-python-structures/actions/workflows/tests.yml/badge.svg)](https://github.com/Quantum-0/custom-python-structures/actions/workflows/tests.yml)
[![Black](https://github.com/Quantum-0/custom-python-structures/actions/workflows/black.yml/badge.svg)](https://github.com/Quantum-0/custom-python-structures/actions/workflows/black.yml)
[![MyPy](https://github.com/Quantum-0/custom-python-structures/actions/workflows/mypy.yml/badge.svg)](https://github.com/Quantum-0/custom-python-structures/actions/workflows/mypy.yml)
[![PyLint](https://github.com/Quantum-0/custom-python-structures/actions/workflows/lint.yml/badge.svg)](https://github.com/Quantum-0/custom-python-structures/actions/workflows/lint.yml)
[![Coverage](https://github.com/Quantum-0/custom-python-structures/actions/workflows/coverage.yml/badge.svg)](https://github.com/Quantum-0/custom-python-structures/actions/workflows/coverage.yml)

# Structures

## LoopList

Sequence structure, implements list with feature, where indexes go by cycle.

### Default `List` sequence:
```mermaid
graph LR
    A --> B --> C --> D --> E
```

### Loop List
```mermaid
graph LR
    A --> B --> C --> D --> E --> A
```

So, if in default `List` you try to get 5th element from that sequence, you'll get `IndexError: list index out of range`.

But with LoopList 5th element will be **A** again, 6th = **B**, 9th = **E**, 10th = **A** again and etc.

Same with negative indexes: -1th element == **E**, -2th = **D**, -5th == **A**, -6th == **E** again and etc.

That structure also supports slices, but currently without step, so you can try get slise `[-2:7]` and that will return list `[D,E,A,B,C,D,E,A,B]`.

Idea was inspired by [**Josephus Problem**](https://en.wikipedia.org/wiki/Josephus_problem) and with that structure solution will look like that:
```python
ring = LoopList(range(1, n + 1))
while len(ring) > 1:
    ring.rotate(k - 1)
    del ring[0]
return ring[0]
```

## Matrix

That structure implements matrix interface on nested list.

### UML
```mermaid
classDiagram
        Matrix <|-- NumericMatrix
        Matrix <|-- BitMatrix
        class Matrix{
            -_values : [[T]]
            -_width : int
            -_height : int
            +width int
            +height int
            +size tuple
            +is_square bool
            +rotated_... Matrix
            +mirrored_... Matrix
            +main_diagonal [T]
            +__getitem__(key)
            +__setitem__(key, value)
            +generate(...)
            +from_nested_list(list of lists)
            +from_joined_lists(w, h, list)
            +from_lists(*lists)
            +input_matrix(...)
            +transpose()
            +get_minor(i, j)
        }
        class NumericMatrix{
            +zero_matrix(n, [m]) NumericMatrix
            +identity(n) NumericMatrix
            +trace : int or float
            +determinant : int or float
            +__add__(other)
            +__sub__(other)
            +__mul__(other)
            +__div__(other)
            +__invert__()
            +__neg__()
        }
        class BitMatrix{
            +zero_matrix(n, [m]) BitMatrix
            +identity(n) BitMatrix
            +__and__(other)
            +__or__(other)
            +__xor__(other)
            +__sub__(other)
            +__neg__()
        }
        MatrixIterator --o Matrix
        MatrixIterator: matrix
        MatrixIterator: WALKTHROW_TYPE
        MatrixIterator: +__init__(Matrix)
        MatrixIterator: +__iter__()
        MatrixIterator: +__next__()
```

### Usage

```python
# Creating examples
m1 = NumericMatrix.from_joined_lists(3, values=range(9))
m2 = Matrix(2, 2, [['A', 'B'], ['C', 'D']])
m3 = NumericMatrix.zero_matrix(size=4)
m4 = BitMatrix.from_lists([True, False], [False, True])

# Comparing
assert m1 != m2 # Supports compating between matrix
assert m2 == [['A', 'B'], ['C', 'D']] # And directly with nested list

# Math
# A, B : NumericMatrix
assert A + B == B + A == C
assert A += B
assert A == C
# Same with subtraction
# Multiplication implements matrix multiplication, so:
assert A * B != B * A
# Trace and Determinant:
A.trace
A.determinant

# Transformations
A.transponate()
assert A.rotated_clockwise.rotated_counterclockwise == A
assert A.mirrored_horizontaly.mirrored_horizontaly == A

# Bit operations - implements bit logic between elements
# A, B : BitMatrix
assert A & B == B & A
assert A | B == B | A
assert A ^ B == B ^ A
assert -(-A) == A

# Indexing and slicing
A[0,0]
A[1,:]
A[:,2]
A[4:7,2:4]
```