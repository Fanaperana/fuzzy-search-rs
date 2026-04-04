//! # fzfrs - Fuzzy Search Algorithm Implementation
//!
//! A comprehensive, educational implementation of fuzzy search algorithms.
//! This library provides tools for approximate string matching using the
//! Levenshtein distance algorithm.
//!
//! ---
//!
//! ## What is fuzzy search?
//!
//! A normal search requires an **exact** match: searching for `"aple"` in a list
//! won't find `"apple"`.  Fuzzy search relaxes that requirement — it finds strings
//! that are *close enough*, even when they contain typos or small differences.
//!
//! The key idea is to measure how many **edits** (insert / delete / substitute a
//! single character) it takes to turn one string into another.  Fewer edits means
//! a better match.
//!
//! ---
//!
//! ## Learning path — where to start
//!
//! If you are new to this topic, read the types in this order:
//!
//! 1. **[`LevenshteinDistance::compute`]** — the classic full-matrix
//!    Wagner-Fischer algorithm.  Every cell, every loop, every step is annotated.
//!    Start here to understand the core idea.
//!
//! 2. **[`LevenshteinDistance::compute_optimized`]** — keeps only two rows of the
//!    matrix at once.  A natural next step once you have grasped the full version.
//!
//! 3. **[`LevenshteinDistance::similarity`]** — converts the raw distance into a
//!    0.0–1.0 score humans can reason about.
//!
//! 4. **[`LevenshteinWithOperations`]** — builds the full list of edits by
//!    *backtracking* through the completed matrix.  Great for visualising what the
//!    algorithm actually does.
//!
//! 5. **[`FuzzySearcher`]** — the high-level search interface that brings
//!    everything together and adds practical features like case-insensitivity and
//!    result ranking.
//!
//! 6. **[`LevenshteinDistance::compute_fast`]** — the performance-tuned variant
//!    with an ASCII byte path and smaller DP cells.  Read this last, once you
//!    understand what it is optimising.
//!
//! ---
//!
//! ## Quick Start
//!
//! ```rust
//! use fzfrs::{FuzzySearcher, LevenshteinDistance};
//!
//! // Calculate edit distance between two strings
//! let distance = LevenshteinDistance::compute("kitten", "sitting");
//! assert_eq!(distance, 3);
//!
//! // Get similarity score (0.0 to 1.0)
//! let similarity = LevenshteinDistance::similarity("apple", "aple");
//! assert!(similarity >= 0.8);
//!
//! // Search through candidates
//! let searcher = FuzzySearcher::new(0.6);
//! let candidates = vec!["apple", "application", "banana"];
//! let results = searcher.search("aple", &candidates);
//! ```
//!
//! ---
//!
//! ## Algorithm Overview
//!
//! The Levenshtein distance between two strings is the minimum number of
//! single-character edits (insertions, deletions, or substitutions) required
//! to change one string into the other.
//!
//! This implementation uses the Wagner-Fischer dynamic programming algorithm
//! with an optional memory-optimized variant that uses only O(min(m,n)) space.

use std::cmp::min;
use std::fmt;

// ============================================================================
// LEVENSHTEIN DISTANCE - Core Algorithm Implementation
//
// If you are reading this for the first time, focus on `compute` first.
// It uses a full 2-D matrix so that every value is visible and easy to trace.
// The later methods (`compute_optimized`, `compute_fast`) produce identical
// results using less memory or fewer CPU cycles, but are harder to follow
// before you understand the basics.
// ============================================================================

/// Core implementation of the Levenshtein distance algorithm.
///
/// The Levenshtein distance (also known as **edit distance**) is the minimum
/// number of single-character edits needed to turn one string into another.
/// Think of it as the "cost" of fixing a typo.
///
/// ## The three operations
///
/// | Operation    | Description           | Example                    |
/// |--------------|-----------------------|----------------------------|
/// | Insert       | Add a character       | `cat` → `cart` (insert r)  |
/// | Delete       | Remove a character    | `cart` → `cat` (delete r)  |
/// | Substitute   | Replace a character   | `cat` → `bat` (sub c → b)  |
///
/// ## Methods at a glance
///
/// | Method | Best for |
/// |--------|----------|
/// | [`compute`](Self::compute) | Learning — shows the full matrix |
/// | [`compute_optimized`](Self::compute_optimized) | Long strings — saves memory |
/// | [`compute_fast`](Self::compute_fast) | Production — fastest path |
/// | [`similarity`](Self::similarity) | Human-readable 0.0–1.0 score |
///
/// ## Example
///
/// ```rust
/// use fzfrs::LevenshteinDistance;
///
/// let distance = LevenshteinDistance::compute("kitten", "sitting");
/// assert_eq!(distance, 3); // k→s, e→i, insert g
///
/// let similarity = LevenshteinDistance::similarity("apple", "apple");
/// assert_eq!(similarity, 1.0); // Identical strings
/// ```
pub struct LevenshteinDistance;

impl LevenshteinDistance {
    /// Computes the Levenshtein distance between two strings.
    ///
    /// Uses the standard Wagner-Fischer algorithm with O(m × n) time and space.
    ///
    /// ## Algorithm (Pseudo Code)
    ///
    /// ```text
    /// FUNCTION levenshtein_distance(source, target):
    ///     m = length(source)
    ///     n = length(target)
    ///     
    ///     IF m == 0: RETURN n
    ///     IF n == 0: RETURN m
    ///     
    ///     matrix = new Matrix[m+1][n+1]
    ///     
    ///     // Initialize first column and row
    ///     FOR i FROM 0 TO m: matrix[i][0] = i
    ///     FOR j FROM 0 TO n: matrix[0][j] = j
    ///     
    ///     // Fill the matrix
    ///     FOR i FROM 1 TO m:
    ///         FOR j FROM 1 TO n:
    ///             cost = IF source[i-1] == target[j-1] THEN 0 ELSE 1
    ///             matrix[i][j] = MIN(
    ///                 matrix[i-1][j] + 1,      // deletion
    ///                 matrix[i][j-1] + 1,      // insertion
    ///                 matrix[i-1][j-1] + cost  // substitution
    ///             )
    ///     
    ///     RETURN matrix[m][n]
    /// ```
    ///
    /// ## Parameters
    ///
    /// * `source` - The source string
    /// * `target` - The target string to compare against
    ///
    /// ## Returns
    ///
    /// The minimum number of edits required to transform source into target.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::LevenshteinDistance;
    ///
    /// assert_eq!(LevenshteinDistance::compute("", "abc"), 3);
    /// assert_eq!(LevenshteinDistance::compute("abc", ""), 3);
    /// assert_eq!(LevenshteinDistance::compute("abc", "abc"), 0);
    /// assert_eq!(LevenshteinDistance::compute("kitten", "sitting"), 3);
    /// ```
    pub fn compute(source: &str, target: &str) -> usize {
        // Collect characters so we can index into them by position.
        // Rust strings are UTF-8 and cannot be indexed directly by integer,
        // so we build a Vec<char> first.  This is the clearest approach for
        // learning purposes (compute_fast avoids this allocation for ASCII).
        let source_chars: Vec<char> = source.chars().collect();
        let target_chars: Vec<char> = target.chars().collect();

        let m = source_chars.len(); // number of characters in source
        let n = target_chars.len(); // number of characters in target

        // ── Edge cases ───────────────────────────────────────────────────────
        // If one string is empty, the only way to transform it into the other
        // is to insert (or delete) every character — so the distance equals
        // the other string's length.
        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }

        // ── Build the DP matrix ──────────────────────────────────────────────
        //
        // We create a 2-D grid of size (m+1) × (n+1).
        //
        // What does matrix[i][j] mean?
        //   The minimum number of edits to transform source[0..i] into target[0..j].
        //   In other words, the sub-problem answer for the first i chars of source
        //   and the first j chars of target.
        //
        // The +1 in each dimension makes room for the "empty prefix" case (i=0 or j=0),
        // which is our base case.
        let mut matrix = vec![vec![0usize; n + 1]; m + 1];

        // ── STEP 1: Initialize the first column ──────────────────────────────
        // matrix[i][0] = i  means: turning source[0..i] into an empty string
        // costs exactly i operations (delete every character).
        for i in 0..=m {
            matrix[i][0] = i;
        }

        // ── STEP 2: Initialize the first row ─────────────────────────────────
        // matrix[0][j] = j  means: turning an empty string into target[0..j]
        // costs exactly j operations (insert every character).
        for j in 0..=n {
            matrix[0][j] = j;
        }

        // ── STEP 3: Fill in the rest of the matrix ───────────────────────────
        //
        // For each (i, j) we ask: what is the cheapest way to turn
        // source[0..i] into target[0..j]?
        //
        // There are three choices, and we take the cheapest:
        //
        //   Deletion:     remove source[i-1].  Cost = matrix[i-1][j] + 1
        //                 (solve the sub-problem without source[i-1], then delete it)
        //
        //   Insertion:    insert target[j-1] at the end of source[0..i].
        //                 Cost = matrix[i][j-1] + 1
        //                 (solve the sub-problem without target[j-1], then insert it)
        //
        //   Substitution: if source[i-1] == target[j-1] the characters already match
        //                 and cost is 0; otherwise replace one with the other, cost 1.
        //                 Either way: matrix[i-1][j-1] + cost
        //                 (solve the sub-problem for both prefixes shortened by one)
        for i in 1..=m {
            for j in 1..=n {
                // cost = 0: characters already match, no substitution needed.
                // cost = 1: characters differ, a substitution is required.
                let cost = if source_chars[i - 1] == target_chars[j - 1] {
                    0
                } else {
                    1
                };

                let deletion = matrix[i - 1][j] + 1; // remove source[i-1]
                let insertion = matrix[i][j - 1] + 1; // insert target[j-1]
                let substitution = matrix[i - 1][j - 1] + cost; // replace/match

                matrix[i][j] = min(deletion, min(insertion, substitution));
            }
        }

        // ── STEP 4: Read the answer ───────────────────────────────────────────
        // The bottom-right cell holds the edit distance for the full strings.
        matrix[m][n]
    }

    /// Memory-optimized version using only two rows instead of the full matrix.
    ///
    /// **Key insight:** look at the recurrence in [`compute`](Self::compute).
    /// Each cell `matrix[i][j]` only ever reads from the row *directly above*
    /// (`matrix[i-1][...]`) — it never looks further back.  That means once
    /// we finish row `i` we will never need rows `0..i-1` again.
    ///
    /// So instead of keeping the whole matrix in memory, we keep only two rows:
    /// - `previous_row` — the completed row above (what `compute` calls row `i-1`)
    /// - `current_row`  — the row we are currently filling (row `i`)
    ///
    /// After filling `current_row` we swap the two pointers and move to the next row.
    /// Memory usage drops from O(m × n) to O(min(m, n)).
    ///
    /// The results are **identical** to [`compute`](Self::compute).
    /// Useful for comparing very long strings.
    ///
    /// ## Algorithm (Pseudo Code)
    ///
    /// ```text
    /// FUNCTION levenshtein_optimized(source, target):
    ///     m = length(source)
    ///     n = length(target)
    ///     
    ///     IF m == 0: RETURN n
    ///     IF n == 0: RETURN m
    ///     
    ///     // Ensure we use the shorter string for columns (less memory)
    ///     IF m < n: SWAP(source, target); SWAP(m, n)
    ///     
    ///     previous_row = [0, 1, 2, ..., n]
    ///     current_row = new Array[n+1]
    ///     
    ///     FOR i FROM 1 TO m:
    ///         current_row[0] = i
    ///         FOR j FROM 1 TO n:
    ///             cost = IF source[i-1] == target[j-1] THEN 0 ELSE 1
    ///             current_row[j] = MIN(
    ///                 current_row[j-1] + 1,    // insertion
    ///                 previous_row[j] + 1,      // deletion
    ///                 previous_row[j-1] + cost  // substitution
    ///             )
    ///         SWAP(previous_row, current_row)
    ///     
    ///     RETURN previous_row[n]
    /// ```
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::LevenshteinDistance;
    ///
    /// // Same result as compute(), but uses less memory
    /// let distance = LevenshteinDistance::compute_optimized("kitten", "sitting");
    /// assert_eq!(distance, 3);
    /// ```
    pub fn compute_optimized(source: &str, target: &str) -> usize {
        let source_chars: Vec<char> = source.chars().collect();
        let target_chars: Vec<char> = target.chars().collect();

        let m = source_chars.len();
        let n = target_chars.len();

        // Handle edge cases
        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }

        // Minor trick: put the *shorter* string in the column dimension so both
        // rows are as small as possible.  The distance is symmetric, so this is safe.
        let (source_chars, target_chars, m, n) = if m < n {
            (target_chars, source_chars, n, m)
        } else {
            (source_chars, target_chars, m, n)
        };

        // previous_row starts as [0, 1, 2, ..., n], which is the same as the
        // first row of the full matrix: transforming "" into target[0..j] costs j.
        let mut previous_row: Vec<usize> = (0..=n).collect();
        let mut current_row: Vec<usize> = vec![0; n + 1];

        for i in 1..=m {
            // The leftmost cell of each row is i: transforming source[0..i] into ""
            // costs i deletions.
            current_row[0] = i;

            for j in 1..=n {
                let cost = if source_chars[i - 1] == target_chars[j - 1] {
                    0
                } else {
                    1
                };

                current_row[j] = min(
                    current_row[j - 1] + 1, // Insertion  (came from left in same row)
                    min(
                        previous_row[j] + 1,        // Deletion   (came from above)
                        previous_row[j - 1] + cost, // Substitution/match (came from diagonal)
                    ),
                );
            }

            // This row is complete.  Make it the "previous" row and reuse
            // the old previous_row buffer as the next current_row.
            std::mem::swap(&mut previous_row, &mut current_row);
        }

        // After the final swap, the answer sits in previous_row[n].
        previous_row[n]
    }

    /// Converts an edit distance into a human-readable similarity score.
    ///
    /// Returns a value between `0.0` (completely different) and `1.0` (identical).
    ///
    /// ## Why this formula?
    ///
    /// Raw edit distance is hard to reason about without knowing string lengths.
    /// The distance between `"a"` and `"b"` is 1, and so is the distance between
    /// `"abcdef"` and `"abcdeg"` — but the second pair is far more similar.
    ///
    /// Dividing distance by the maximum possible distance (= `max_length`, i.e.
    /// rewriting every character) normalises it to the 0–1 range, then we flip
    /// it so that 1.0 means identical and 0.0 means completely different.
    ///
    /// ## Formula
    ///
    /// ```text
    /// similarity = 1.0 - (distance / max_length)
    /// ```
    ///
    /// A score ≥ 0.8 is generally considered a good match for typo correction.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::LevenshteinDistance;
    ///
    /// assert_eq!(LevenshteinDistance::similarity("abc", "abc"), 1.0);
    /// assert_eq!(LevenshteinDistance::similarity("", ""), 1.0);
    ///
    /// let sim = LevenshteinDistance::similarity("kitten", "sitting");
    /// assert!((sim - 0.571).abs() < 0.01); // ~57.1%
    /// ```
    pub fn similarity(source: &str, target: &str) -> f64 {
        let source_len = source.chars().count();
        let target_len = target.chars().count();
        let max_len = source_len.max(target_len);

        // Two empty strings are considered identical
        if max_len == 0 {
            return 1.0;
        }

        let distance = Self::compute_fast(source, target);

        // Convert distance to similarity score
        1.0 - (distance as f64 / max_len as f64)
    }

    /// Case-insensitive similarity calculation.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::LevenshteinDistance;
    ///
    /// let sim = LevenshteinDistance::similarity_ignore_case("Apple", "APPLE");
    /// assert_eq!(sim, 1.0);
    /// ```
    pub fn similarity_ignore_case(source: &str, target: &str) -> f64 {
        Self::similarity(&source.to_lowercase(), &target.to_lowercase())
    }

    /// Production-ready distance calculation — same result as [`compute`](Self::compute)
    /// but faster, used internally by [`similarity`](Self::similarity) and
    /// [`FuzzySearcher::search`].
    ///
    /// This method selects the best implementation automatically:
    ///
    /// - **ASCII strings** → [`fast_bytes`](Self::fast_bytes): works on raw bytes,
    ///   skipping the `Vec<char>` allocation entirely.
    /// - **Unicode strings** → [`fast_chars`](Self::fast_chars): falls back to
    ///   collecting chars, same as the other methods, but with `u32` cells.
    ///
    /// Both paths use `u32` (4 bytes) per DP cell instead of `usize` (8 bytes on
    /// 64-bit), so the two rolling rows fit into half the cache space, which matters
    /// when the strings are long.
    ///
    /// **Why read this last?** Understanding `compute` and `compute_optimized` first
    /// makes the optimisations here much easier to appreciate.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::LevenshteinDistance;
    ///
    /// assert_eq!(LevenshteinDistance::compute_fast("kitten", "sitting"), 3);
    /// assert_eq!(LevenshteinDistance::compute_fast("", "abc"), 3);
    /// ```
    pub fn compute_fast(source: &str, target: &str) -> usize {
        if source.is_ascii() && target.is_ascii() {
            // ASCII guarantee: every character is exactly one byte, so a byte index
            // is the same as a character index.  We can skip Vec<char> entirely.
            Self::fast_bytes(source.as_bytes(), target.as_bytes())
        } else {
            // Non-ASCII (emoji, accented letters, CJK, …): fall back to char slices.
            let src: Vec<char> = source.chars().collect();
            let tgt: Vec<char> = target.chars().collect();
            Self::fast_chars(&src, &tgt)
        }
    }

    // ── fast_bytes ────────────────────────────────────────────────────────────
    // Two-row DP over raw byte slices — the ASCII-only fast path.
    //
    // Optimisation 1 – no Vec<char> allocation:
    //   ASCII characters are all in the range 0x00–0x7F and each fits in exactly
    //   one byte.  `str::as_bytes()` gives us a &[u8] with no heap allocation.
    //
    // Optimisation 2 – u32 cells (4 bytes) instead of usize (8 bytes on 64-bit):
    //   A string would have to be > 4 billion characters long to overflow u32.
    //   Using u32 means the two rolling rows take half as much memory, which
    //   keeps them in CPU cache longer and improves throughput on medium/long strings.
    fn fast_bytes(src: &[u8], tgt: &[u8]) -> usize {
        let m = src.len();
        let n = tgt.len();
        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }
        // Keep the shorter sequence in the column dimension (smaller row allocation).
        let (src, tgt) = if m < n { (tgt, src) } else { (src, tgt) };
        let (m, n) = (src.len(), tgt.len());

        let mut prev: Vec<u32> = (0..=(n as u32)).collect(); // base row: [0, 1, 2, …, n]
        let mut curr: Vec<u32> = vec![0u32; n + 1]; // scratch buffer, reused each row

        for i in 1..=m {
            curr[0] = i as u32; // cost of deleting all of src[0..i]
            let s = src[i - 1]; // current source byte, hoisted out of the inner loop
            for j in 1..=n {
                // u32::from(bool) gives 0 when characters match, 1 when they differ.
                let cost = u32::from(s != tgt[j - 1]);
                curr[j] = (curr[j - 1] + 1) // insertion
                    .min(prev[j] + 1) // deletion
                    .min(prev[j - 1] + cost); // substitution / match
            }
            std::mem::swap(&mut prev, &mut curr); // roll forward
        }
        prev[n] as usize
    }

    // ── fast_chars ────────────────────────────────────────────────────────────
    // Identical logic to fast_bytes but operates on char slices instead of byte
    // slices.  Used when at least one string contains non-ASCII characters.
    fn fast_chars(src: &[char], tgt: &[char]) -> usize {
        let m = src.len();
        let n = tgt.len();
        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }
        let (src, tgt) = if m < n { (tgt, src) } else { (src, tgt) };
        let (m, n) = (src.len(), tgt.len());

        let mut prev: Vec<u32> = (0..=(n as u32)).collect();
        let mut curr: Vec<u32> = vec![0u32; n + 1];

        for i in 1..=m {
            curr[0] = i as u32;
            let s = src[i - 1];
            for j in 1..=n {
                let cost = u32::from(s != tgt[j - 1]);
                curr[j] = (curr[j - 1] + 1) // insertion
                    .min(prev[j] + 1) // deletion
                    .min(prev[j - 1] + cost); // substitution / match
            }
            std::mem::swap(&mut prev, &mut curr);
        }
        prev[n] as usize
    }
}

// ============================================================================
// EDIT OPERATIONS - Tracking what changes are needed
//
// The methods above tell you *how many* edits are needed.
// This section tells you *which* edits: insert 'r', delete 'k', etc.
//
// The trick is backtracking: after building the full DP matrix we start at the
// bottom-right corner and work backwards, choosing at each step the operation
// that produced the current cell's value.
// ============================================================================

/// Represents a single edit operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditOperation {
    /// Keep the character (no change needed)
    Keep(char),
    /// Insert a character
    Insert(char),
    /// Delete a character
    Delete(char),
    /// Substitute one character for another
    Substitute { from: char, to: char },
}

impl fmt::Display for EditOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EditOperation::Keep(c) => write!(f, "keep '{}'", c),
            EditOperation::Insert(c) => write!(f, "insert '{}'", c),
            EditOperation::Delete(c) => write!(f, "delete '{}'", c),
            EditOperation::Substitute { from, to } => write!(f, "replace '{}' → '{}'", from, to),
        }
    }
}

/// Detailed result of comparing two strings.
#[derive(Debug, Clone)]
pub struct EditResult {
    /// The source string
    pub source: String,
    /// The target string
    pub target: String,
    /// The edit distance
    pub distance: usize,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f64,
    /// The sequence of operations to transform source into target
    pub operations: Vec<EditOperation>,
}

impl EditResult {
    /// Returns a human-readable description of the match quality.
    pub fn quality(&self) -> &'static str {
        match (self.similarity * 100.0) as u32 {
            95..=100 => "Excellent",
            85..=94 => "Very Good",
            70..=84 => "Good",
            50..=69 => "Fair",
            _ => "Poor",
        }
    }
}

impl fmt::Display for EditResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Comparison: '{}' → '{}'", self.source, self.target)?;
        writeln!(f, "Distance: {} operations", self.distance)?;
        writeln!(
            f,
            "Similarity: {:.1}% ({})",
            self.similarity * 100.0,
            self.quality()
        )?;
        if !self.operations.is_empty() {
            writeln!(f, "Operations:")?;
            for (i, op) in self.operations.iter().enumerate() {
                writeln!(f, "  {}. {}", i + 1, op)?;
            }
        }
        Ok(())
    }
}

/// Extended Levenshtein that records the exact sequence of edits.
///
/// Under the hood this works in two phases:
///
/// 1. **Forward pass** — build the same DP matrix as [`LevenshteinDistance::compute`].
/// 2. **Backtracking pass** — start at `matrix[m][n]` and walk backwards,
///    at each step choosing the neighbour cell whose value explains the current
///    cell (deletion came from above, insertion from the left, substitution from
///    the diagonal).  The path we walk defines the edit script.
///
/// This is great for understanding *why* two strings got the score they did.
pub struct LevenshteinWithOperations;

impl LevenshteinWithOperations {
    /// Computes the edit distance and returns detailed information including
    /// the sequence of operations needed.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::LevenshteinWithOperations;
    ///
    /// let result = LevenshteinWithOperations::compute("cat", "cart");
    /// println!("{}", result);
    /// // Shows: insert 'r'
    /// ```
    pub fn compute(source: &str, target: &str) -> EditResult {
        let source_chars: Vec<char> = source.chars().collect();
        let target_chars: Vec<char> = target.chars().collect();

        let m = source_chars.len();
        let n = target_chars.len();

        // Edge cases
        if m == 0 && n == 0 {
            return EditResult {
                source: source.to_string(),
                target: target.to_string(),
                distance: 0,
                similarity: 1.0,
                operations: vec![],
            };
        }

        if m == 0 {
            return EditResult {
                source: source.to_string(),
                target: target.to_string(),
                distance: n,
                similarity: 0.0,
                operations: target_chars
                    .iter()
                    .map(|&c| EditOperation::Insert(c))
                    .collect(),
            };
        }

        if n == 0 {
            return EditResult {
                source: source.to_string(),
                target: target.to_string(),
                distance: m,
                similarity: 0.0,
                operations: source_chars
                    .iter()
                    .map(|&c| EditOperation::Delete(c))
                    .collect(),
            };
        }

        // Phase 1: build the full DP matrix (same as LevenshteinDistance::compute).
        // We need the entire matrix (not just two rows) because we will walk
        // backwards through it in the backtracking phase below.
        let mut matrix = vec![vec![0usize; n + 1]; m + 1];

        for i in 0..=m {
            matrix[i][0] = i; // base case: delete all of source[0..i]
        }
        for j in 0..=n {
            matrix[0][j] = j; // base case: insert all of target[0..j]
        }

        for i in 1..=m {
            for j in 1..=n {
                let cost = if source_chars[i - 1] == target_chars[j - 1] {
                    0 // characters match — no substitution cost
                } else {
                    1 // characters differ — substitution costs 1
                };

                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,                                   // deletion
                    min(matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + cost), // insertion / sub
                );
            }
        }

        // Phase 2: backtrack through the completed matrix to recover the edit path.
        let operations = Self::backtrack(&matrix, &source_chars, &target_chars);

        let distance = matrix[m][n];
        let max_len = m.max(n);
        let similarity = 1.0 - (distance as f64 / max_len as f64);

        EditResult {
            source: source.to_string(),
            target: target.to_string(),
            distance,
            similarity,
            operations,
        }
    }

    // Backtracks through the completed DP matrix to reconstruct the edit script.
    //
    // How backtracking works
    // ──────────────────────
    // We filled matrix[i][j] by choosing the cheapest of three neighbours:
    //
    //   matrix[i-1][j]   + 1   → we deleted source[i-1]       (came from above)
    //   matrix[i][j-1]   + 1   → we inserted target[j-1]      (came from left)
    //   matrix[i-1][j-1] + c   → we substituted (or matched)   (came from diagonal)
    //
    // To recover the path we start at the bottom-right corner (i=m, j=n)
    // and ask "which neighbour produced this cell's value?"  That tells us
    // which operation was chosen.  We move to that neighbour and repeat,
    // collecting operations in reverse order until we reach (0, 0).
    fn backtrack(
        matrix: &[Vec<usize>],
        source_chars: &[char],
        target_chars: &[char],
    ) -> Vec<EditOperation> {
        let mut operations = Vec::new();
        let mut i = source_chars.len(); // start at bottom-right
        let mut j = target_chars.len();

        while i > 0 || j > 0 {
            if i > 0 && j > 0 && source_chars[i - 1] == target_chars[j - 1] {
                // Characters match → came from the diagonal at no cost.
                // We record a Keep so callers can see the full alignment if they want.
                operations.push(EditOperation::Keep(source_chars[i - 1]));
                i -= 1;
                j -= 1;
            } else if i > 0 && j > 0 && matrix[i][j] == matrix[i - 1][j - 1] + 1 {
                // Substitution: the diagonal cell + 1 produced this cell.
                // source[i-1] was replaced with target[j-1].
                operations.push(EditOperation::Substitute {
                    from: source_chars[i - 1],
                    to: target_chars[j - 1],
                });
                i -= 1;
                j -= 1;
            } else if j > 0 && matrix[i][j] == matrix[i][j - 1] + 1 {
                // Insertion: the left cell + 1 produced this cell.
                // target[j-1] was inserted into source.
                operations.push(EditOperation::Insert(target_chars[j - 1]));
                j -= 1;
            } else if i > 0 && matrix[i][j] == matrix[i - 1][j] + 1 {
                // Deletion: the cell above + 1 produced this cell.
                // source[i-1] was deleted.
                operations.push(EditOperation::Delete(source_chars[i - 1]));
                i -= 1;
            } else {
                // Guard: should not be reached with a valid matrix.
                break;
            }
        }

        // We collected operations back-to-front; reverse to get source → target order.
        operations.reverse();

        // Remove Keep entries for a concise output (only actual edits are shown).
        operations
            .into_iter()
            .filter(|op| !matches!(op, EditOperation::Keep(_)))
            .collect()
    }
}

// ============================================================================
// FUZZY SEARCHER - High-level search interface
//
// Everything above is low-level machinery.  FuzzySearcher wraps it into a
// practical API: compare a query against a list of candidates, filter out
// poor matches, sort by quality, and optionally cap the result count.
// ============================================================================

/// A single result returned by [`FuzzySearcher::search`].
#[derive(Debug, Clone)]
pub struct MatchResult {
    /// The matched text
    pub text: String,
    /// Similarity score (0.0 to 1.0)
    pub score: f64,
    /// The original index in the candidates list
    pub index: usize,
}

impl MatchResult {
    /// Returns a human-readable quality description.
    pub fn quality(&self) -> &'static str {
        match (self.score * 100.0) as u32 {
            95..=100 => "Excellent",
            85..=94 => "Very Good",
            70..=84 => "Good",
            50..=69 => "Fair",
            _ => "Poor",
        }
    }
}

/// High-level fuzzy search engine.
///
/// Compares a query string against a list of candidates using
/// [`LevenshteinDistance::similarity`] and returns matches that score above a
/// configurable threshold, sorted best-first.
///
/// ## Choosing a threshold
///
/// | Threshold | Meaning |
/// |-----------|--------|
/// | `1.0` | Exact matches only |
/// | `0.8` | Allows ~1 typo in a 5-character word |
/// | `0.6` | Loose matching — good default for autocomplete |
/// | `0.3` | Very loose — catches heavily misspelled words |
///
/// Builder methods ([`case_insensitive`](Self::case_insensitive),
/// [`max_results`](Self::max_results)) use the **method-chaining** pattern:
/// each call returns `Self` so you can chain configuration on one line.
#[derive(Debug, Clone)]
pub struct FuzzySearcher {
    /// Minimum similarity threshold (0.0 to 1.0).
    /// Candidates below this score are excluded from results.
    threshold: f64,
    /// When `true`, both query and candidate are lowercased before comparison,
    /// so `"Apple"` and `"apple"` score 1.0.
    case_insensitive: bool,
    /// Cap on the number of results returned.  `None` means unlimited.
    max_results: Option<usize>,
}

impl Default for FuzzySearcher {
    fn default() -> Self {
        Self {
            threshold: 0.6,
            case_insensitive: true,
            max_results: None,
        }
    }
}

impl FuzzySearcher {
    /// Creates a new FuzzySearcher with the given similarity threshold.
    ///
    /// ## Parameters
    ///
    /// * `threshold` - Minimum similarity score (0.0 to 1.0) for a match
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::FuzzySearcher;
    ///
    /// let searcher = FuzzySearcher::new(0.7); // 70% similarity required
    /// ```
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Enables or disables case-insensitive matching.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::FuzzySearcher;
    ///
    /// let searcher = FuzzySearcher::new(0.6)
    ///     .case_insensitive(false); // Case-sensitive matching
    /// ```
    pub fn case_insensitive(mut self, value: bool) -> Self {
        self.case_insensitive = value;
        self
    }

    /// Sets the maximum number of results to return.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::FuzzySearcher;
    ///
    /// let searcher = FuzzySearcher::new(0.6)
    ///     .max_results(10); // Return at most 10 matches
    /// ```
    pub fn max_results(mut self, limit: usize) -> Self {
        self.max_results = Some(limit);
        self
    }

    /// Searches for matches in the given candidates.
    ///
    /// Returns a list of matches sorted by score (best first).
    ///
    /// ## Algorithm (Pseudo Code)
    ///
    /// ```text
    /// FUNCTION fuzzy_search(query, candidates, threshold):
    ///     results = []
    ///     
    ///     FOR EACH (index, candidate) IN candidates:
    ///         score = similarity(query, candidate)
    ///         
    ///         IF score >= threshold:
    ///             ADD (candidate, score, index) TO results
    ///     
    ///     SORT results BY score DESCENDING
    ///     RETURN results
    /// ```
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::FuzzySearcher;
    ///
    /// let searcher = FuzzySearcher::new(0.6);
    /// let candidates = vec!["apple", "application", "banana"];
    /// let results = searcher.search("aple", &candidates);
    ///
    /// for result in results {
    ///     println!("{}: {:.1}%", result.text, result.score * 100.0);
    /// }
    /// ```
    pub fn search<S: AsRef<str>>(&self, query: &str, candidates: &[S]) -> Vec<MatchResult> {
        let query_normalized = if self.case_insensitive {
            query.to_lowercase()
        } else {
            query.to_string()
        };

        let mut results: Vec<MatchResult> = candidates
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| {
                let candidate_str = candidate.as_ref();
                let candidate_normalized = if self.case_insensitive {
                    candidate_str.to_lowercase()
                } else {
                    candidate_str.to_string()
                };

                // ── Early-exit optimisation ───────────────────────────────
                //
                // The edit distance between two strings is *at least* the
                // absolute difference of their lengths.  Why?  Even if every
                // character in the shorter string matches a character in the
                // longer one perfectly, you still need at least |len_a - len_b|
                // insertions or deletions to handle the extra characters.
                //
                // Therefore the best possible similarity score is:
                //   1.0 - |len_a - len_b| / max(len_a, len_b)
                //
                // If even this *best-case* score is below our threshold, we
                // can skip the DP entirely — no amount of shared characters
                // can save this candidate.
                //
                // This is only exact for ASCII (where byte length == char length).
                // For non-ASCII we skip the check rather than risk a false negative.
                if query_normalized.is_ascii() && candidate_normalized.is_ascii() {
                    let qlen = query_normalized.len();
                    let clen = candidate_normalized.len();
                    let max_len = qlen.max(clen);
                    if max_len > 0 {
                        let len_diff = qlen.abs_diff(clen);
                        let max_similarity = 1.0 - (len_diff as f64 / max_len as f64);
                        if max_similarity < self.threshold {
                            return None; // cannot possibly meet the threshold
                        }
                    }
                }

                let score =
                    LevenshteinDistance::similarity(&query_normalized, &candidate_normalized);

                if score >= self.threshold {
                    Some(MatchResult {
                        text: candidate_str.to_string(),
                        score,
                        index,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by score descending (best matches first)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply max_results limit if set
        if let Some(limit) = self.max_results {
            results.truncate(limit);
        }

        results
    }

    /// Finds the single best match for a query.
    ///
    /// Returns None if no match meets the threshold.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::FuzzySearcher;
    ///
    /// let searcher = FuzzySearcher::new(0.6);
    /// let candidates = vec!["apple", "banana", "orange"];
    ///
    /// if let Some(best) = searcher.find_best("aple", &candidates) {
    ///     println!("Best match: {} ({:.1}%)", best.text, best.score * 100.0);
    /// }
    /// ```
    pub fn find_best<S: AsRef<str>>(&self, query: &str, candidates: &[S]) -> Option<MatchResult> {
        self.search(query, candidates).into_iter().next()
    }

    /// Checks if a query matches a single candidate.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use fzfrs::FuzzySearcher;
    ///
    /// let searcher = FuzzySearcher::new(0.7);
    ///
    /// assert!(searcher.matches("apple", "aple"));
    /// assert!(!searcher.matches("apple", "banana"));
    /// ```
    pub fn matches(&self, candidate: &str, query: &str) -> bool {
        let score = if self.case_insensitive {
            LevenshteinDistance::similarity_ignore_case(query, candidate)
        } else {
            LevenshteinDistance::similarity(query, candidate)
        };

        score >= self.threshold
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_basic() {
        assert_eq!(LevenshteinDistance::compute("", ""), 0);
        assert_eq!(LevenshteinDistance::compute("abc", ""), 3);
        assert_eq!(LevenshteinDistance::compute("", "abc"), 3);
        assert_eq!(LevenshteinDistance::compute("abc", "abc"), 0);
    }

    #[test]
    fn test_levenshtein_single_operations() {
        // Single insertion
        assert_eq!(LevenshteinDistance::compute("cat", "cart"), 1);
        // Single deletion
        assert_eq!(LevenshteinDistance::compute("cart", "cat"), 1);
        // Single substitution
        assert_eq!(LevenshteinDistance::compute("cat", "bat"), 1);
    }

    #[test]
    fn test_levenshtein_complex() {
        assert_eq!(LevenshteinDistance::compute("kitten", "sitting"), 3);
        assert_eq!(LevenshteinDistance::compute("saturday", "sunday"), 3);
        assert_eq!(LevenshteinDistance::compute("algorithm", "altruistic"), 6);
    }

    #[test]
    fn test_optimized_matches_standard() {
        let test_pairs = [
            ("", ""),
            ("abc", ""),
            ("", "abc"),
            ("kitten", "sitting"),
            ("algorithm", "altruistic"),
            ("hello world", "hello word"),
        ];

        for (s1, s2) in test_pairs {
            assert_eq!(
                LevenshteinDistance::compute(s1, s2),
                LevenshteinDistance::compute_optimized(s1, s2),
                "Mismatch for '{}' vs '{}'",
                s1,
                s2
            );
        }
    }

    #[test]
    fn test_similarity() {
        assert_eq!(LevenshteinDistance::similarity("abc", "abc"), 1.0);
        assert_eq!(LevenshteinDistance::similarity("", ""), 1.0);
        assert_eq!(LevenshteinDistance::similarity("abc", "xyz"), 0.0);

        let sim = LevenshteinDistance::similarity("kitten", "sitting");
        assert!((sim - 0.571).abs() < 0.01);
    }

    #[test]
    fn test_case_insensitive() {
        assert_eq!(
            LevenshteinDistance::similarity_ignore_case("Apple", "APPLE"),
            1.0
        );
        assert_eq!(
            LevenshteinDistance::similarity_ignore_case("Hello", "hello"),
            1.0
        );
    }

    #[test]
    fn test_fuzzy_searcher() {
        let searcher = FuzzySearcher::new(0.6);
        let candidates = vec!["apple", "application", "applet", "banana"];

        let results = searcher.search("aple", &candidates);

        assert!(!results.is_empty());
        assert_eq!(results[0].text, "apple"); // Best match
    }

    #[test]
    fn test_fuzzy_searcher_case_insensitive() {
        let searcher = FuzzySearcher::new(0.8).case_insensitive(true);
        let candidates = vec!["Apple", "APPLE", "apple"];

        let results = searcher.search("apple", &candidates);

        assert_eq!(results.len(), 3);
        for result in results {
            assert_eq!(result.score, 1.0);
        }
    }

    #[test]
    fn test_fuzzy_searcher_max_results() {
        let searcher = FuzzySearcher::new(0.5).max_results(2);
        let candidates = vec!["apple", "application", "applet", "app"];

        let results = searcher.search("app", &candidates);

        assert!(results.len() <= 2);
    }

    #[test]
    fn test_edit_operations() {
        let result = LevenshteinWithOperations::compute("cat", "cart");
        assert_eq!(result.distance, 1);
        assert_eq!(result.operations.len(), 1);
        assert!(matches!(result.operations[0], EditOperation::Insert('r')));
    }

    #[test]
    fn test_unicode() {
        assert_eq!(LevenshteinDistance::compute("café", "cafe"), 1);
        assert_eq!(LevenshteinDistance::compute("日本", "日本語"), 1);
        assert_eq!(LevenshteinDistance::compute("🎉", "🎊"), 1);
    }
}
