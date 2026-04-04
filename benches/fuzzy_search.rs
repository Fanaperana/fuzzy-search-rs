//! Benchmark: Levenshtein implementation comparison
//!
//! Three algorithms are pitted against each other across five input tiers:
//!
//! | Tier   | Approx length | What it stresses                                   |
//! |--------|---------------|----------------------------------------------------|
//! | tiny   | ≤10 chars     | Function-call overhead dominates                   |
//! | short  | ~25 chars     | Matrix stays small – space gap not yet visible     |
//! | medium | ~60 chars     | Cache pressure begins to matter                    |
//! | long   | ~190 chars    | Full O(m×n) vs two-row gap becomes clearly visible |
//! | stress | ~270 chars    | Worst-case: entirely reversed strings, no matches  |
//!
//! Run:           cargo bench
//! HTML reports:  target/criterion/<group>/report/index.html

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fzfrs::LevenshteinDistance;

// ── Test inputs ───────────────────────────────────────────────────────────────
//
// Pairs are chosen to require real DP work (not nearly-identical strings)
// so that all cells of the matrix must be filled.  Each tuple is:
//   (source, target, human-readable-label)

const INPUTS: &[(&str, &str, &str)] = &[
    // tiny: classic textbook example
    (
        "kitten",
        "sitting",
        "tiny (6 vs 7 chars)",
    ),
    // short: two plausible but distinct phrases
    (
        "fuzzy search algorithm",
        "approximate string match",
        "short (22 vs 24 chars)",
    ),
    // medium: full sentences with shared structure but different words
    (
        "the quick brown fox jumps over the lazy dog near the old river bank",
        "a slow red cat leaps across the muddy stream under a grey winter sky",
        "medium (66 vs 68 chars)",
    ),
    // long: two paragraphs about related-but-different topics
    (
        "dynamic programming is a method for solving complex problems \
         by breaking them down into simpler overlapping subproblems \
         and storing the results of each subproblem to avoid redundant work",
        "divide and conquer is an algorithm design paradigm that splits \
         a problem into independent subproblems solves them recursively \
         and combines their solutions but does not cache intermediate results",
        "long (182 vs 191 chars)",
    ),
    // stress: maximum editing required – alphabet repeated forwards vs backwards.
    // Every cell of the DP matrix is forced to compute a non-trivial value.
    (
        "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz\
         abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz\
         abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz\
         abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxy",
        "zyxwvutsrqponmlkjihgfedcbazyxwvutsrqponmlkjihgfedcba\
         zyxwvutsrqponmlkjihgfedcbazyxwvutsrqponmlkjihgfedcba\
         zyxwvutsrqponmlkjihgfedcbazyxwvutsrqponmlkjihgfedcba\
         zyxwvutsrqponmlkjihgfedcbazyxwvutsrqponmlkjihgfedcb",
        "stress (270 vs 268 chars)",
    ),
];

// ── Head-to-head comparison ───────────────────────────────────────────────────
//
// Every implementation runs on the exact same (source, target) pair so
// Criterion's report puts them side-by-side with a clear speedup label.
//
// What each algorithm does differently:
//
//  compute           – full (m+1)×(n+1) matrix, usize cells (8 bytes on x86-64).
//                      Easiest to understand; highest memory use.
//
//  compute_optimized – two-row rolling buffer instead of the full matrix.
//                      Identical results, O(min(m,n)) space.
//                      Cells are still usize so cache density is the same.
//
//  compute_fast      – two-row rolling buffer + u32 cells (4 bytes, 2× cache
//                      density) + ASCII byte path (no Vec<char> allocation).
//                      Fastest; same results.
//
// At "tiny" the three will look similar because overhead dominates.
// By "stress" compute_fast should be clearly faster than compute.

fn bench_levenshtein_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("levenshtein");

    for &(src, tgt, label) in INPUTS {
        group.bench_with_input(
            BenchmarkId::new("1_compute", label),
            &(src, tgt),
            |b, (s, t)| b.iter(|| LevenshteinDistance::compute(black_box(s), black_box(t))),
        );

        group.bench_with_input(
            BenchmarkId::new("2_compute_optimized", label),
            &(src, tgt),
            |b, (s, t)| {
                b.iter(|| LevenshteinDistance::compute_optimized(black_box(s), black_box(t)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("3_compute_fast", label),
            &(src, tgt),
            |b, (s, t)| {
                b.iter(|| LevenshteinDistance::compute_fast(black_box(s), black_box(t)))
            },
        );
    }

    group.finish();
}

// ── Entry point ───────────────────────────────────────────────────────────────

criterion_group!(levenshtein, bench_levenshtein_comparison);
criterion_main!(levenshtein);
