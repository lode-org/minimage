//! Reduction of a periodic pair list to one minimum-image pair.
//!
//! vesin enumerates periodic images, so a particle can appear as its
//! own neighbour through an image, and one neighbour can arrive through
//! several images at once. The minimum-image convention admits each
//! ordered pair once and never the self pair.

use crate::Error;

/// Drop self images and collapse duplicate `(i, j)` rows.
///
/// Input is a flat `n * 2` list of signed indices. Output is sorted
/// unique pairs with `i != j`, written into `out` (capacity at least
/// `pairs.len()`). Returns the number of kept pairs.
pub fn reduce_pairs(pairs: &[[i32; 2]], out: &mut Vec<[i32; 2]>) -> Result<usize, Error> {
    out.clear();
    if pairs.is_empty() {
        return Ok(0);
    }
    out.extend(pairs.iter().copied().filter(|p| p[0] != p[1]));
    out.sort_unstable();
    out.dedup();
    Ok(out.len())
}

/// Same reduction over a packed `n * 2` C buffer.
///
/// `out` must have room for `n` pairs (`2 * n` ints). Writes the kept
/// count into `out_n`.
pub fn reduce_pairs_packed(pairs: &[i32], out: &mut [i32], out_n: &mut usize) -> Result<(), Error> {
    if pairs.len() % 2 != 0 {
        return Err(Error::BufferSize);
    }
    let n = pairs.len() / 2;
    if out.len() < pairs.len() {
        return Err(Error::BufferSize);
    }
    let mut tmp = Vec::with_capacity(n);
    for k in 0..n {
        let i = pairs[2 * k];
        let j = pairs[2 * k + 1];
        if i != j {
            tmp.push([i, j]);
        }
    }
    tmp.sort_unstable();
    tmp.dedup();
    *out_n = tmp.len();
    for (k, pair) in tmp.iter().enumerate() {
        out[2 * k] = pair[0];
        out[2 * k + 1] = pair[1];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_self_and_duplicate_images() {
        let pairs = [[0, 1], [1, 0], [0, 0], [0, 1], [2, 2], [2, 3]];
        let mut out = Vec::new();
        let n = reduce_pairs(&pairs, &mut out).unwrap();
        assert_eq!(n, 3);
        assert_eq!(out, vec![[0, 1], [1, 0], [2, 3]]);
    }

    #[test]
    fn packed_matches_rows() {
        let packed = [0, 1, 0, 0, 0, 1, 2, 3];
        let mut out = [0; 8];
        let mut n = 99;
        reduce_pairs_packed(&packed, &mut out, &mut n).unwrap();
        assert_eq!(n, 2);
        assert_eq!(&out[..4], &[0, 1, 2, 3]);
    }
}
