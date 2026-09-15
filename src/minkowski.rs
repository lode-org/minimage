//! Minkowski reduction of a 3D lattice basis.
//!
//! Nguyen and Stehlé, ACM Trans. Algorithms 5, 46 (2009)
//! <https://doi.org/10.1145/1597036.1597050>: after the basis is
//! Minkowski-reduced, the shortest vector from a point to the lattice
//! sits among the 27 combinations of `{-1,0,1}` in that basis.
//! GROMACS `correct_box` / LAMMPS tilt limits do not give that
//! guarantee. The extra work matters for cutoff-free k-NN, where the
//! k-th neighbour can sit near the covering radius (a hex-prism body
//! diagonal is the witness). MD pair forces never see that regime.

use crate::cell::Cell;
use crate::Error;

const TOL: f64 = 1e-12;
const MAX_IT: usize = 10_000;

/// True when the three columns of H satisfy the 3D Minkowski
/// inequalities (Jaber; Nguyen–Stehlé).
pub fn is_minkowski_reduced(cell: &Cell) -> bool {
    let a = cell.a();
    let b = cell.b();
    let c = cell.c();
    let na = n2(a);
    let nb = n2(b);
    let nc = n2(c);
    if !(na <= nb + TOL && nb <= nc + TOL) {
        return false;
    }
    // ASE / Jaber inequalities. rhs indexes |a|,|b|,|c|.
    let rhs_idx = [0, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2];
    let norms = [na.sqrt(), nb.sqrt(), nc.sqrt()];
    for (i, row) in [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, -1.0, 0.0],
        [1.0, 0.0, -1.0],
        [0.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ]
    .iter()
    .enumerate()
    {
        let v = lin(*row, a, b, c);
        if n2(v).sqrt() + TOL < norms[rhs_idx[i]] {
            return false;
        }
    }
    true
}

/// Unimodular Minkowski reduction. Same Cartesian lattice, new basis
/// with `|a| <= |b| <= |c|`.
pub fn minkowski_reduce(cell: &Cell) -> Result<Cell, Error> {
    if is_minkowski_reduced(cell) {
        return Ok(*cell);
    }
    let mut basis = [cell.a(), cell.b(), cell.c()];
    for _ in 0..MAX_IT {
        sort_by_length(&mut basis);
        let (head, tail) = basis.split_at_mut(1);
        gauss_pair(&mut head[0], &mut tail[0]);
        subtract_plane(&mut basis);
        sort_by_length(&mut basis);
        let trial = Cell::from_vectors(basis[0], basis[1], basis[2], cell.origin())?;
        if is_minkowski_reduced(&trial) {
            return Ok(trial);
        }
        if n2(basis[2]) + TOL >= n2(basis[1]) && n2(basis[1]) + TOL >= n2(basis[0]) {
            // Gauss + plane subtract stalled; accept if the 12
            // inequalities hold after one more sort.
            if is_minkowski_reduced(&trial) {
                return Ok(trial);
            }
        }
    }
    Cell::from_vectors(basis[0], basis[1], basis[2], cell.origin())
}

fn n2(v: [f64; 3]) -> f64 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

fn add(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    [u[0] + v[0], u[1] + v[1], u[2] + v[2]]
}

fn scale(s: f64, v: [f64; 3]) -> [f64; 3] {
    [s * v[0], s * v[1], s * v[2]]
}

fn lin(c: [f64; 3], a: [f64; 3], b: [f64; 3], d: [f64; 3]) -> [f64; 3] {
    add(add(scale(c[0], a), scale(c[1], b)), scale(c[2], d))
}

fn dot(u: [f64; 3], v: [f64; 3]) -> f64 {
    u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
}

fn sort_by_length(basis: &mut [[f64; 3]; 3]) {
    if n2(basis[0]) > n2(basis[1]) {
        basis.swap(0, 1);
    }
    if n2(basis[1]) > n2(basis[2]) {
        basis.swap(1, 2);
    }
    if n2(basis[0]) > n2(basis[1]) {
        basis.swap(0, 1);
    }
}

/// Lagrange/Gauss reduction of two vectors.
fn gauss_pair(u: &mut [f64; 3], v: &mut [f64; 3]) {
    for _ in 0..MAX_IT {
        if n2(*v) + TOL < n2(*u) {
            std::mem::swap(u, v);
        }
        let uu = n2(*u);
        if uu < TOL {
            return;
        }
        let n = (dot(*u, *v) / uu).round() as i32;
        if n == 0 {
            return;
        }
        *v = add(*v, scale(-f64::from(n), *u));
    }
}

/// Subtract the nearest integer combination of the two short vectors
/// from the third (Nguyen–Stehlé closest-vector step in the plane).
fn subtract_plane(basis: &mut [[f64; 3]; 3]) {
    let u = basis[0];
    let v = basis[1];
    let mut w = basis[2];
    let uu = n2(u);
    let vv = n2(v);
    if uu < TOL || vv < TOL {
        return;
    }
    // One Gauss-style pass in the plane of u,v, then a 3x3 stencil
    // around that image. Enough for 3D Minkowski in practice.
    let nu = (dot(w, u) / uu).round();
    w = add(w, scale(-nu, u));
    let nv = (dot(w, v) / vv).round();
    w = add(w, scale(-nv, v));
    let mut best = w;
    let mut best2 = n2(w);
    for i in -1..=1 {
        for j in -1..=1 {
            if i == 0 && j == 0 {
                continue;
            }
            let t = add(add(w, scale(f64::from(i), u)), scale(f64::from(j), v));
            let t2 = n2(t);
            if t2 < best2 {
                best2 = t2;
                best = t;
            }
        }
    }
    basis[2] = best;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ortho_is_minkowski() {
        let c = Cell::ortho(10.0, 11.0, 12.0).unwrap();
        assert!(is_minkowski_reduced(&c));
    }

    #[test]
    fn hex_prism_is_minkowski() {
        let c = Cell::from_vectors(
            [10.0, 0.0, 0.0],
            [5.0, 8.660254037844386, 0.0],
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 0.0],
        )
        .unwrap();
        assert!(is_minkowski_reduced(&c));
    }

    #[test]
    fn extreme_skew_is_not_minkowski_until_reduced() {
        let c = Cell::from_vectors(
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        )
        .unwrap();
        assert!(!is_minkowski_reduced(&c));
        let r = minkowski_reduce(&c).unwrap();
        assert!(is_minkowski_reduced(&r));
    }
}
