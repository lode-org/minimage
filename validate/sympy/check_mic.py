#!/usr/bin/env python3
"""Exact checks for the orthorhombic wrap and one Delone step.

Floor splitting is the identity floor(n + t) = n + floor(t) for integer
n. On the piece x = n - 1/2 + t the residual is t - 1/2 - floor(t), which
is t - 1/2 once 0 <= t < 1. The right endpoint x = n + 1/2 is the next
integer and lands on -1/2, so +L/2 ties to -L/2.
"""

import json
import sys

from sympy import (
    Integer,
    Matrix,
    Rational,
    floor,
    simplify,
    symbols,
)


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    print(json.dumps({"agent": "sympy", "choice": "reject"}))
    raise SystemExit(1)


def wrap(d, length):
    return d - length * floor(d / length + Rational(1, 2))


def check_floor_split() -> None:
    n = symbols("n", integer=True)
    t = symbols("t", real=True)
    # SymPy's floor splits an integer addend. The piece is built from that.
    if floor(n + t) != n + floor(t):
        fail("floor(n + t) did not split")
    x = n - Rational(1, 2) + t
    residual = simplify(x - (n + floor(t)))
    if residual != t - Rational(1, 2) - floor(t):
        fail(f"piece residual {residual}")
    # 0 <= t < 1 means floor(t) = 0, so the residual is t - 1/2
    # and therefore lies in [-1/2, 1/2).
    left = n - Rational(1, 2)
    if floor(left + Rational(1, 2)) != n:
        fail("left endpoint floor")
    if simplify(left - n) != Rational(-1, 2):
        fail("left endpoint residual")
    right = n + Rational(1, 2)
    if floor(right + Rational(1, 2)) != n + 1:
        fail("half tie floor")
    if simplify(right - (n + 1)) != Rational(-1, 2):
        fail("half tie residual")


def check_wrap_values() -> None:
    cases = [
        (10, 15, -5),
        (10, 5, -5),
        (10, -5, -5),
        (10, 25, -5),
        (10, 0, 0),
        (11, -18, 4),
        (12, 12, 0),
        (12, 6, -6),
    ]
    for length, disp, expect in cases:
        got = wrap(Integer(disp), Integer(length))
        if got != expect:
            fail(f"wrap({disp}, {length}) = {got}, expected {expect}")


def selling_step(cols, i: int, j: int):
    out = [c.copy() for c in cols]
    step = out[i]
    for k in range(4):
        if k != i and k != j:
            out[k] = out[k] + step
    out[i] = -step
    return out


def check_selling() -> None:
    names = symbols("a0:3 b0:3 c0:3")
    a = Matrix(names[0:3])
    b = Matrix(names[3:6])
    c = Matrix(names[6:9])
    d = -(a + b + c)
    before = [a, b, c, d]
    after = selling_step(before, 0, 1)

    def total(cols):
        acc = Matrix.zeros(3, 1)
        for col in cols:
            acc += col
        return acc

    if simplify(total(after) - total(before)) != Matrix.zeros(3, 1):
        fail("Delone step moved the sum")
    # New (a', b', c') in the old basis: -a, b, c + a.
    change = Matrix([[-1, 0, 1], [0, 1, 0], [0, 0, 1]])
    if change.det() != -1:
        fail(f"Delone determinant {change.det()}")
    dot = simplify(a.dot(b))
    old_sq = sum(simplify(v.dot(v)) for v in before)
    new_sq = sum(simplify(v.dot(v)) for v in after)
    # ||v||^2 summed over the four vectors drops by twice the pair dot.
    # The negative sum of the six Selling scalars therefore drops as well
    # (Andrews, Bernstein, and Sauter, Acta Cryst. A75, 2019).
    if simplify(old_sq - new_sq - 2 * dot) != 0:
        fail("squared length did not drop by twice the pair dot")


def shell_min(y, coeffs) -> Rational:
    best = None
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                p = i * coeffs[0] + j * coeffs[1] + k * coeffs[2]
                d2 = sum((y[t] - p[t]) ** 2 for t in range(3))
                best = d2 if best is None else min(best, d2)
    return best


def check_shells() -> None:
    ortho = shell_min(
        Matrix([25, -18, 3]),
        [Matrix([10, 0, 0]), Matrix([0, 11, 0]), Matrix([0, 0, 12])],
    )
    if ortho != 50:
        fail(f"ortho shell {ortho}")
    a = Matrix([1, 0, 0])
    b = Matrix([Rational(99, 100), Rational(1, 100), 0])
    c = Matrix([0, 0, 1])
    y = 2 * (a - b)
    skew = shell_min(y, [a, b, c])
    if skew != 0:
        fail(f"skew shell {skew}")
    a = Matrix([1, 0, 0])
    b = Matrix([1 - Rational(1, 128), Rational(1, 128), 0])
    c = Matrix([0, 0, 1])
    dyadic = shell_min(2 * (a - b), [a, b, c])
    if dyadic != 0:
        fail(f"dyadic skew shell {dyadic}")


def main() -> None:
    check_floor_split()
    check_wrap_values()
    check_selling()
    check_shells()
    print(json.dumps({"agent": "sympy", "choice": "accept"}))


if __name__ == "__main__":
    main()
