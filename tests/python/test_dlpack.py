import gc

import numpy as np
import pytest

import minimage

HEX = ([10.0, 0.0, 0.0], [5.0, 8.660254037844386, 0.0], [0.0, 0.0, 10.0])


class Producer:
    """A DLPack producer that is not numpy, the way torch or jax hand one over."""

    def __init__(self, array, device=(1, 0)):
        self._array = array
        self._device = device

    def __dlpack__(self, **kwargs):
        return self._array.__dlpack__(**kwargs)

    def __dlpack_device__(self):
        return self._device


def points(n, seed):
    return np.random.default_rng(seed).random((n, 3)) * 30.0 - 10.0


def test_arrays_in_give_arrays_out_and_match_lists():
    cell = minimage.Cell.from_vectors(*HEX)
    ps, qs = points(257, 1), points(257, 2)
    p = [0.5, 1.5, 2.5]
    many = cell.dist2_many(p, qs)
    assert isinstance(many, np.ndarray) and many.dtype == np.float64
    np.testing.assert_array_equal(many, cell.dist2_many(p, qs.tolist()))
    pairs = cell.dist2_pairs(ps, qs)
    assert isinstance(pairs, np.ndarray)
    np.testing.assert_array_equal(pairs, cell.dist2_pairs(ps.tolist(), qs.tolist()))
    wrapped = cell.wrap_many(qs - ps)
    assert wrapped.shape == (257, 3)
    np.testing.assert_array_equal(wrapped, np.array(cell.wrap_many((qs - ps).tolist())))


def test_strided_fortran_float32_and_readonly_inputs():
    cell = minimage.Cell.from_vectors(*HEX)
    base = points(400, 3)
    p = [1.0, 2.0, 3.0]
    want = np.array(cell.dist2_many(p, base[::2].tolist()))
    np.testing.assert_array_equal(cell.dist2_many(p, base[::2]), want)
    np.testing.assert_array_equal(cell.dist2_many(p, np.asfortranarray(base[::2])), want)
    frozen = base[::2].copy()
    frozen.setflags(write=False)
    np.testing.assert_array_equal(cell.dist2_many(p, frozen), want)
    single = base[::2].astype(np.float32)
    np.testing.assert_allclose(
        cell.dist2_many(p, single),
        cell.dist2_many(p, single.astype(np.float64).tolist()),
        rtol=1e-12,
    )


def test_any_dlpack_producer_and_host_only():
    cell = minimage.Cell.ortho(10.0, 11.0, 12.0)
    qs = points(64, 4)
    p = [0.2, 0.0, 0.0]
    np.testing.assert_array_equal(cell.dist2_many(p, Producer(qs)), cell.dist2_many(p, qs))
    with pytest.raises(TypeError):
        cell.dist2_many(p, Producer(qs, device=(2, 0)))
    with pytest.raises(ValueError):
        cell.dist2_many(p, np.zeros((5, 2)))


def test_fixed_point_through_arrays():
    for cell in (minimage.Cell.ortho(10.0, 11.0, 12.0), minimage.Cell.from_vectors(*HEX)):
        ps, qs = points(129, 5), points(129, 6)
        fp, fq = cell.fixed_many(ps), cell.fixed_many(qs)
        assert fp.dtype == np.uint64 and fp.shape == (129, 3)
        assert list(fp[7]) == cell.fixed(ps[7].tolist())
        np.testing.assert_allclose(cell.dist2_pairs_fixed(fp, fq), cell.dist2_pairs(ps, qs), rtol=1e-12)
        np.testing.assert_allclose(
            cell.dist2_many_fixed(fp[0], fq), cell.dist2_many(ps[0], qs), rtol=1e-12
        )
        assert cell.dist2_fixed(fp[3], fq[3]) == pytest.approx(cell.dist2(ps[3], qs[3]), rel=1e-12)
        np.testing.assert_array_equal(
            cell.dist2_pairs_fixed(fp.tolist(), fq.tolist()), cell.dist2_pairs_fixed(fp, fq)
        )


def test_results_outlive_inputs_and_cell():
    cell = minimage.Cell.from_vectors(*HEX)
    qs = points(1000, 7)
    out = cell.dist2_many([0.0, 0.0, 0.0], qs)
    want = out.copy()
    del cell, qs
    gc.collect()
    np.testing.assert_array_equal(out, want)
    out[0] = -1.0
    assert out[0] == -1.0
