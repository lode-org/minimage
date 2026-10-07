//! Minimum image on [Burn](https://burn.dev) tensors.
//!
//! Burn picks the backend from the [`Device`] at run time, so one code
//! path serves the CPU (`flex`, `cpu`) and GPUs (`wgpu`, `vulkan`,
//! `metal`, `cuda`, `rocm`); enable the backend features the application
//! needs. Every operation here is a matrix product or an element-wise
//! map, which Burn's `fusion` decorator joins into one kernel per batch.
//!
//! Two paths:
//!
//! - [`Lattice::dist2_pairs`], [`Lattice::dist2_many`], [`Lattice::wrap`]:
//!   Cartesian rows in, the engine wrap `H (s - round(s))` with
//!   `s = Hinv d`, as `minimage` computes it on the CPU.
//! - [`Lattice::fractions`] folds positions into `[0, 1)` once, on the
//!   CPU in double precision; [`Lattice::dist2_pairs_fractional`] and
//!   friends then wrap differences of those fractions. For `|ds| < 1`,
//!   `ds - round(ds)` is exact in every float width (Sterbenz), so a
//!   half- or single-precision device keeps an exact wrap and loses only
//!   the rounding of `ds` and of the product with `H`. This is the
//!   symmetric residue of Ozaki Scheme II (Ozaki, Uchino, and Imamura,
//!   arXiv:2504.08009, 2025) in floating point, and the explicit reduced
//!   precision of kernel_float (Heldens et al., Netherlands eScience
//!   Center).
//!
//! Burn rounds halves to even, so a fraction exactly one half apart may
//! wrap to `+1/2` where `minimage` keeps `-1/2`; the squared distance is
//! the same.

use burn_tensor::{DType, Device, Tensor, TensorData};
use minimage::Cell;

/// One cell's `H` and `Hinv`, transposed for row vectors, on a device.
#[derive(Clone, Debug)]
pub struct Lattice {
    /// Rows `a, b, c`: a fractional row times this is Cartesian.
    h_t: Tensor<2>,
    /// Cartesian row times this is fractional.
    hinv_t: Tensor<2>,
    origin: Tensor<2>,
    device: Device,
    dtype: DType,
}

impl Lattice {
    /// `cell` on `device` in `dtype` (`F64`, `F32`, `F16`, `BF16`).
    pub fn new(cell: &Cell, device: &Device, dtype: DType) -> Self {
        let h = cell.h();
        let hinv = cell.hinv();
        // Columns of H are rows of H^T.
        let rows = |m: [[f64; 3]; 3]| -> Vec<f64> { m.iter().flatten().copied().collect() };
        let o = cell.origin();
        Self {
            h_t: matrix(rows(h), device, dtype),
            hinv_t: matrix(rows(hinv), device, dtype),
            origin: Tensor::<2>::from_data(TensorData::new(o.to_vec(), [1, 3]), (device, dtype)),
            device: device.clone(),
            dtype,
        }
    }

    /// Packed `[x, y, z]` rows as an `(n, 3)` tensor on this lattice's
    /// device and dtype.
    pub fn rows(&self, rows: &[[f64; 3]]) -> Tensor<2> {
        rows_tensor(rows, &self.device, self.dtype)
    }

    /// Fractions in `[0, 1)` of each position, folded on the CPU in double
    /// precision ([`Cell::fixed`]) and uploaded in this lattice's dtype.
    /// A far position keeps its digits; folding on a single-precision
    /// device would not.
    pub fn fractions(&self, cell: &Cell, rows: &[[f64; 3]]) -> Tensor<2> {
        let mut fixed = vec![[0u64; 3]; rows.len()];
        minimage::fixed_many(cell, rows, &mut fixed).expect("one output per row");
        let scale = 1.0 / 18_446_744_073_709_551_616.0;
        let flat: Vec<f64> = fixed
            .iter()
            .flatten()
            .map(|&u| (u >> 12) as f64 * (scale * 4096.0))
            .collect();
        Tensor::<2>::from_data(
            TensorData::new(flat, [rows.len(), 3]),
            (&self.device, self.dtype),
        )
    }

    /// Fractions of `(n, 3)` Cartesian rows folded on the device.
    pub fn fractional(&self, rs: Tensor<2>) -> Tensor<2> {
        let s = rs.sub(self.origin.clone()).matmul(self.hinv_t.clone());
        s.clone().sub(s.floor())
    }

    /// Engine wrap of `(n, 3)` difference rows.
    pub fn wrap(&self, diffs: Tensor<2>) -> Tensor<2> {
        let s = diffs.matmul(self.hinv_t.clone());
        s.clone().sub(s.round()).matmul(self.h_t.clone())
    }

    /// Squared engine-wrap distance of each row pair.
    pub fn dist2_pairs(&self, ps: Tensor<2>, qs: Tensor<2>) -> Tensor<1> {
        norm2(self.wrap(qs.sub(ps)))
    }

    /// Squared engine-wrap distance from the `(1, 3)` row `p` to each row.
    pub fn dist2_many(&self, p: Tensor<2>, qs: Tensor<2>) -> Tensor<1> {
        norm2(self.wrap(qs.sub(p)))
    }

    /// Cartesian displacement between fraction rows from
    /// [`Self::fractions`] or [`Self::fractional`]. The wrap is exact.
    pub fn displacement_fractional(&self, sp: Tensor<2>, sq: Tensor<2>) -> Tensor<2> {
        let ds = sq.sub(sp);
        ds.clone().sub(ds.round()).matmul(self.h_t.clone())
    }

    /// Squared distance of each fraction row pair.
    pub fn dist2_pairs_fractional(&self, sp: Tensor<2>, sq: Tensor<2>) -> Tensor<1> {
        norm2(self.displacement_fractional(sp, sq))
    }

    /// Squared distance from the `(1, 3)` fraction row `sp` to each row.
    pub fn dist2_many_fractional(&self, sp: Tensor<2>, sq: Tensor<2>) -> Tensor<1> {
        norm2(self.displacement_fractional(sp, sq))
    }
}

fn matrix(values: Vec<f64>, device: &Device, dtype: DType) -> Tensor<2> {
    Tensor::<2>::from_data(TensorData::new(values, [3, 3]), (device, dtype))
}

/// Packed rows as an `(n, 3)` tensor.
pub fn rows_tensor(rows: &[[f64; 3]], device: &Device, dtype: DType) -> Tensor<2> {
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    Tensor::<2>::from_data(TensorData::new(flat, [rows.len(), 3]), (device, dtype))
}

/// A rank-one tensor read back as doubles.
pub fn to_vec(t: Tensor<1>) -> Vec<f64> {
    t.into_data()
        .convert::<f64>()
        .try_to_vec::<f64>()
        .expect("a float tensor converts to f64")
}

fn norm2(x: Tensor<2>) -> Tensor<1> {
    let n = x.dims()[0];
    x.powi_scalar(2).sum_dim(1).reshape([n])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cells() -> Vec<Cell> {
        vec![
            Cell::ortho(10.0, 11.0, 12.0).unwrap(),
            Cell::from_vectors(
                [10.0, 0.0, 0.0],
                [5.0, 8.660254037844386, 0.0],
                [0.0, 0.0, 10.0],
                [1.0, -2.0, 0.5],
            )
            .unwrap(),
            Cell::from_vectors(
                [9.0, 3.0, 2.0],
                [1.0, 9.0, -2.0],
                [-1.5, 2.0, 9.5],
                [0.0; 3],
            )
            .unwrap(),
        ]
    }

    fn points(n: usize, seed: u64, span: f64) -> Vec<[f64; 3]> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                let mut p = [0.0; 3];
                for e in &mut p {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    *e = ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * span;
                }
                p
            })
            .collect()
    }

    fn worst(got: &[f64], want: &[f64]) -> f64 {
        got.iter()
            .zip(want)
            .map(|(g, w)| (g - w).abs() / (1.0 + w.abs()))
            .fold(0.0, f64::max)
    }

    #[test]
    fn double_precision_matches_the_cpu_wrap() {
        let device = Device::flex();
        for cell in cells() {
            let lat = Lattice::new(&cell, &device, DType::F64);
            let ps = points(200, 0x1234_5678, 40.0);
            let qs = points(200, 0x9abc_def0, 40.0);
            let want: Vec<f64> = ps
                .iter()
                .zip(&qs)
                .map(|(p, q)| cell.dist2(*p, *q))
                .collect();
            let got = to_vec(lat.dist2_pairs(lat.rows(&ps), lat.rows(&qs)));
            assert!(
                worst(&got, &want) < 1e-12,
                "float path {}",
                worst(&got, &want)
            );
            let frac = to_vec(
                lat.dist2_pairs_fractional(lat.fractions(&cell, &ps), lat.fractions(&cell, &qs)),
            );
            assert!(
                worst(&frac, &want) < 1e-12,
                "fraction path {}",
                worst(&frac, &want)
            );
            let p0 = lat.rows(&ps[..1]);
            let many = to_vec(lat.dist2_many(p0, lat.rows(&qs)));
            let want_many: Vec<f64> = qs.iter().map(|q| cell.dist2(ps[0], *q)).collect();
            assert!(worst(&many, &want_many) < 1e-12);
        }
    }

    #[test]
    fn single_precision_fractions_keep_far_positions() {
        let device = Device::flex();
        for cell in cells() {
            let lat = Lattice::new(&cell, &device, DType::F32);
            // Positions thousands of cells from the origin.
            let ps: Vec<[f64; 3]> = points(200, 0x2468, 40.0)
                .iter()
                .map(|p| [p[0] + 3.0e4, p[1] - 2.0e4, p[2] + 1.0e4])
                .collect();
            let qs = points(200, 0x1357, 40.0);
            let want: Vec<f64> = ps
                .iter()
                .zip(&qs)
                .map(|(p, q)| cell.dist2(*p, *q))
                .collect();
            let frac = to_vec(
                lat.dist2_pairs_fractional(lat.fractions(&cell, &ps), lat.fractions(&cell, &qs)),
            );
            let direct = to_vec(lat.dist2_pairs(lat.rows(&ps), lat.rows(&qs)));
            assert!(
                worst(&frac, &want) < 2e-5,
                "fractions {}",
                worst(&frac, &want)
            );
            assert!(worst(&direct, &want) > worst(&frac, &want));
        }
    }

    /// The same kernels through wgpu. With no GPU, Mesa's lavapipe is a
    /// CPU-type Vulkan adapter, which checks the generated shaders.
    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_fractions_match_the_cpu_wrap() {
        let device = Device::wgpu(burn_tensor::DeviceKind::Cpu);
        for cell in cells() {
            let lat = Lattice::new(&cell, &device, DType::F32);
            let ps = points(256, 0x1111, 40.0);
            let qs = points(256, 0x2222, 40.0);
            let want: Vec<f64> = ps
                .iter()
                .zip(&qs)
                .map(|(p, q)| cell.dist2(*p, *q))
                .collect();
            let frac = to_vec(
                lat.dist2_pairs_fractional(lat.fractions(&cell, &ps), lat.fractions(&cell, &qs)),
            );
            assert!(
                worst(&frac, &want) < 2e-5,
                "wgpu fractions {}",
                worst(&frac, &want)
            );
            let direct = to_vec(lat.dist2_pairs(lat.rows(&ps), lat.rows(&qs)));
            assert!(
                worst(&direct, &want) < 2e-4,
                "wgpu float path {}",
                worst(&direct, &want)
            );
        }
    }

    #[test]
    fn device_fold_and_wrap_match_the_cpu() {
        let device = Device::flex();
        for cell in cells() {
            let lat = Lattice::new(&cell, &device, DType::F64);
            let rs = points(64, 0xfeed, 30.0);
            let s = lat
                .fractional(lat.rows(&rs))
                .into_data()
                .convert::<f64>()
                .try_to_vec::<f64>()
                .unwrap();
            for (k, r) in rs.iter().enumerate() {
                let want = cell.fractional(*r);
                for a in 0..3 {
                    let d = (s[3 * k + a] - want[a]).abs();
                    assert!(d.min(1.0 - d) < 1e-12, "{} {}", s[3 * k + a], want[a]);
                }
            }
            let diffs = points(64, 0xbeef, 50.0);
            let w = lat
                .wrap(lat.rows(&diffs))
                .into_data()
                .convert::<f64>()
                .try_to_vec::<f64>()
                .unwrap();
            for (k, d) in diffs.iter().enumerate() {
                let want = cell.displacement([0.0; 3], *d);
                let got = [w[3 * k], w[3 * k + 1], w[3 * k + 2]];
                let n2 = |v: [f64; 3]| v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                assert!((n2(got) - n2(want)).abs() < 1e-10 * (1.0 + n2(want)));
            }
        }
    }
}
