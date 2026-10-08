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


def test_euclidean_batches_match_the_per_pair_call():
    for cell in (minimage.Cell.from_vectors(*HEX),
                 minimage.Cell.from_vectors([1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.0, 0.0, 1.0])):
        ps, qs = points(203, 8), points(203, 9)
        pairs = cell.dist2_euclidean_pairs(ps, qs)
        assert isinstance(pairs, np.ndarray)
        want = [cell.dist2_euclidean(p, q) for p, q in zip(ps.tolist(), qs.tolist())]
        np.testing.assert_array_equal(pairs, want)
        many = cell.dist2_euclidean_many(ps[0], qs)
        np.testing.assert_array_equal(many, [cell.dist2_euclidean(ps[0].tolist(), q) for q in qs.tolist()])
        assert cell.dist2_euclidean_pairs(ps.tolist(), qs.tolist()) == want


def test_fixed32_through_arrays():
    for cell in (minimage.Cell.ortho(10.0, 11.0, 12.0), minimage.Cell.from_vectors(*HEX)):
        ps, qs = points(77, 10), points(77, 11)
        fp, fq = cell.fixed32_many(ps), cell.fixed32_many(qs)
        assert fp.dtype == np.uint32 and fp.shape == (77, 3)
        assert list(fp[5]) == cell.fixed32(ps[5].tolist())
        got = cell.dist2_pairs_fixed32(fp, fq)
        np.testing.assert_allclose(got, cell.dist2_pairs(ps, qs), rtol=1e-7, atol=1e-7)
        np.testing.assert_array_equal(got, [cell.dist2_fixed32(a, b) for a, b in zip(fp.tolist(), fq.tolist())])
        np.testing.assert_array_equal(cell.dist2_many_fixed32(fp[0], fq), [cell.dist2_fixed32(fp[0].tolist(), b) for b in fq.tolist()])


def test_warm_start_across_frames():
    cell = minimage.Cell.from_vectors([1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.0, 0.0, 1.0])
    rng = np.random.default_rng(12)
    ps = rng.random((300, 3)) * 5.0
    qs = ps + (rng.random((300, 3)) - 0.5) * 6.0
    d2, images = cell.dist2_euclidean_pairs_warm(ps, qs)
    assert images.dtype == np.int32 and images.shape == (300, 3)
    want = cell.dist2_euclidean_pairs(ps, qs)
    np.testing.assert_allclose(d2, want, rtol=1e-11, atol=1e-13)
    qs = qs + (rng.random((300, 3)) - 0.5) * 0.002
    d2, images = cell.dist2_euclidean_pairs_warm(ps, qs, images)
    np.testing.assert_allclose(d2, cell.dist2_euclidean_pairs(ps, qs), rtol=1e-11, atol=1e-13)
