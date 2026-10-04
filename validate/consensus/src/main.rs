//! Settle the three validator ballots, and check this crate against the
//! same fixtures.
//!
//! DeGroot with an empty trust row listens to every voter equally
//! (ljos-consensus). Susceptibility 1 is plain DeGroot, not an anchored
//! Friedkin–Johnsen step. Unanimous `accept` is the only settlement
//! that keeps the accept share at 1.

use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use ljos_consensus::{settle, Ballot};
use minimage::Cell;

fn field(text: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let rest = text.get(text.find(&pat)? + pat.len()..)?;
    let rest = rest.trim_start().strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn read_ballot(path: &Path) -> Result<Ballot, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let agent = field(&text, "agent").ok_or_else(|| format!("{}: no agent", path.display()))?;
    let choice = field(&text, "choice").ok_or_else(|| format!("{}: no choice", path.display()))?;
    Ok(Ballot { agent, choice })
}

fn check_crate() -> Result<(), String> {
    let ortho = Cell::ortho(10.0, 11.0, 12.0).map_err(|e| e.to_string())?;
    let got = ortho.displacement([0.0, 0.0, 0.0], [25.0, -18.0, 3.0]);
    let expect = [-5.0, 4.0, 3.0];
    for i in 0..3 {
        if (got[i] - expect[i]).abs() > 1e-9 {
            return Err(format!("ortho displacement {got:?}"));
        }
    }
    let tie = Cell::ortho(10.0, 10.0, 10.0).map_err(|e| e.to_string())?;
    let half = tie.displacement([0.0, 0.0, 0.0], [5.0, -5.0, 15.0]);
    let half_expect = [-5.0, -5.0, -5.0];
    for i in 0..3 {
        if (half[i] - half_expect[i]).abs() > 1e-9 {
            return Err(format!("half tie {half:?}"));
        }
    }
    let skew = Cell::from_vectors(
        [1.0, 0.0, 0.0],
        [1.0 - 1.0 / 128.0, 1.0 / 128.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    )
    .map_err(|e| e.to_string())?;
    let y = [1.0 / 64.0, -1.0 / 64.0, 0.0];
    let d2 = skew.dist2_euclidean([0.0, 0.0, 0.0], y);
    if d2 > 1e-18 {
        return Err(format!("dyadic lattice vector residual {d2}"));
    }
    Ok(())
}

fn main() -> ExitCode {
    let dir = env::args()
        .nth(1)
        .unwrap_or_else(|| "validate/out".to_string());
    let dir = Path::new(&dir);
    if let Err(err) = check_crate() {
        eprintln!("{err}");
        return ExitCode::from(1);
    }
    let mut ballots = Vec::new();
    for name in ["sympy.json", "sollya.json", "lean.json"] {
        match read_ballot(&dir.join(name)) {
            Ok(ballot) => ballots.push(ballot),
            Err(err) => {
                eprintln!("{err}");
                return ExitCode::from(1);
            }
        }
    }
    // Empty trust: each voter hears every voter, itself included.
    let outcome = settle(&ballots, &[], 0.5, 1.0, 200, 1e-9);
    let accept = outcome
        .options
        .iter()
        .zip(&outcome.shares)
        .find(|(opt, _)| opt.as_str() == "accept")
        .map(|(_, share)| *share)
        .unwrap_or(0.0);
    eprintln!(
        "settled={} rounds={} accept={accept:.3} polarization={:.3e} disagreement={:.3e}",
        outcome.settled, outcome.rounds, outcome.polarization, outcome.disagreement
    );
    for ballot in &ballots {
        eprintln!("  {} -> {}", ballot.agent, ballot.choice);
    }
    if outcome.settled && (accept - 1.0).abs() < 1e-9 {
        ExitCode::SUCCESS
    } else {
        eprintln!("accept share is not 1");
        ExitCode::from(1)
    }
}
