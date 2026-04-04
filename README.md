# fuzzly - Fuzzy Search Algorithm Implementation

<p align="center">
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" /></a>
    <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-%23000000.svg?logo=rust&logoColor=white" alt="Rust" /></a>
    <a href="https://github.com/Fanaperana/fuzzy-search-rs/releases"><img src="https://img.shields.io/github/v/release/Fanaperana/fuzzy-search-rs" alt="GitHub release" /></a>
    <a href="https://fanaperana.github.io/fuzzy-search-rs/"><img src="https://img.shields.io/badge/docs-live-orange" alt="Docs" /></a>
    <a href="https://github.com/Fanaperana/fuzzy-search-rs/actions/workflows/deploy-pages.yml"><img src="https://github.com/Fanaperana/fuzzy-search-rs/actions/workflows/deploy-pages.yml/badge.svg" alt="Deploy GitHub Pages" /></a>
    <a href="https://github.com/Fanaperana/fuzzy-search-rs/stargazers"><img src="https://img.shields.io/github/stars/Fanaperana/fuzzy-search-rs?style=social" alt="GitHub stars" /></a>
</p>

<p align="center">
  <img src="docs/Fuzzly_logo.png" width="620" alt="fuzzly — fuzzy search algorithm in Rust" />
</p>

> ⭐ **If fuzzly helped you learn**, please [leave a star on GitHub](https://github.com/Fanaperana/fuzzy-search-rs)! It helps others discover the project and keeps the motivation going.

A **comprehensive, educational implementation** of fuzzy search algorithms in Rust with an interactive Levenshtein visualizer, benchmarked implementations, and heavily annotated source code.

Quick links: [Docs site](https://fanaperana.github.io/fuzzy-search-rs/) · [First release](https://github.com/Fanaperana/fuzzy-search-rs/releases/tag/v0.1.0) · [Discussions](https://github.com/Fanaperana/fuzzy-search-rs/discussions)

## Why fuzzly?

- Learn fuzzy search from a real, readable Rust implementation
- Compare `compute`, `compute_optimized`, and `compute_fast` side-by-side
- Explore an interactive docs site instead of reading source alone
- Use it as a reference for typo-tolerant search, ranking, and edit distance

## Quick Start

Add fuzzly directly from GitHub today:

```toml
[dependencies]
fuzzly = { git = "https://github.com/Fanaperana/fuzzy-search-rs" }
```

Minimal example:

```rust
use fuzzly::{FuzzySearcher, LevenshteinDistance};

fn main() {
    let distance = LevenshteinDistance::compute("kitten", "sitting");
    println!("distance: {distance}");

    let searcher = FuzzySearcher::new(0.6);
    let results = searcher.search("aple", &["apple", "application", "banana"]);

    for result in results {
        println!("{}: {:.0}%", result.text, result.score * 100.0);
    }
}
```

Best entry points:

- Read the live docs: [fanaperana.github.io/fuzzy-search-rs](https://fanaperana.github.io/fuzzy-search-rs/)
- Browse the release: [v0.1.0](https://github.com/Fanaperana/fuzzy-search-rs/releases/tag/v0.1.0)
- Join the conversation: [GitHub Discussions](https://github.com/Fanaperana/fuzzy-search-rs/discussions)

## 🎯 What is Fuzzy Search?

Fuzzy search (also known as approximate string matching) finds strings that **approximately match** a pattern, rather than requiring an exact match. This is useful for:

- **Typo tolerance**: Finding "apple" when user types "aple"
- **Autocomplete**: Suggesting completions as users type
- **File finders**: Tools like fzf, Ctrl+P in editors
- **Search engines**: Handling misspellings gracefully

## 📚 Algorithm Overview

This implementation uses the **Levenshtein Distance** (Edit Distance) algorithm as its core, enhanced with scoring and ranking mechanisms.

### Core Concept: Edit Distance

The edit distance between two strings is the **minimum number of single-character operations** needed to transform one string into another:

| Operation | Description | Example |
|-----------|-------------|---------|
| **Insert** | Add a character | `cat` → `cart` (insert 'r') |
| **Delete** | Remove a character | `cart` → `cat` (delete 'r') |
| **Substitute** | Replace a character | `cat` → `bat` (substitute 'c' → 'b') |

### The Algorithm: Wagner-Fischer (Dynamic Programming)

We use dynamic programming to efficiently compute the edit distance.

---

## 🔍 Pseudo Code

### 1. Levenshtein Distance Calculation

```
FUNCTION levenshtein_distance(source, target):
    m = length(source)
    n = length(target)
    
    // Handle edge cases
    IF m == 0: RETURN n
    IF n == 0: RETURN m
    
    // Create matrix of size (m+1) x (n+1)
    matrix = new Matrix[m+1][n+1]
    
    // STEP 1: Initialize first column (deletions from source)
    FOR i FROM 0 TO m:
        matrix[i][0] = i
    
    // STEP 2: Initialize first row (insertions into source)  
    FOR j FROM 0 TO n:
        matrix[0][j] = j
    
    // STEP 3: Fill the matrix
    FOR i FROM 1 TO m:
        FOR j FROM 1 TO n:
            // Cost is 0 if characters match, 1 otherwise
            IF source[i-1] == target[j-1]:
                cost = 0
            ELSE:
                cost = 1
            
            // Take minimum of three operations:
            deletion    = matrix[i-1][j] + 1      // Delete from source
            insertion   = matrix[i][j-1] + 1      // Insert into source
            substitution = matrix[i-1][j-1] + cost // Substitute (or match)
            
            matrix[i][j] = MIN(deletion, insertion, substitution)
    
    // STEP 4: Result is in bottom-right cell
    RETURN matrix[m][n]
```

### 2. Similarity Score (0.0 to 1.0)

```
FUNCTION similarity_score(source, target):
    distance = levenshtein_distance(source, target)
    max_length = MAX(length(source), length(target))
    
    IF max_length == 0:
        RETURN 1.0  // Both empty strings are identical
    
    // Convert distance to similarity (1.0 = identical, 0.0 = completely different)
    RETURN 1.0 - (distance / max_length)
```

### 3. Fuzzy Match with Threshold

```
FUNCTION fuzzy_match(query, candidates, threshold):
    results = empty list
    
    FOR each candidate IN candidates:
        score = similarity_score(query, candidate)
        
        IF score >= threshold:
            ADD (candidate, score) TO results
    
    // Sort by score descending (best matches first)
    SORT results BY score DESCENDING
    
    RETURN results
```

### 4. Memory-Optimized Version (Two-Row Technique)

```
FUNCTION levenshtein_distance_optimized(source, target):
    m = length(source)
    n = length(target)
    
    IF m == 0: RETURN n
    IF n == 0: RETURN m
    
    // Only need two rows instead of full matrix
    previous_row = new Array[n+1]
    current_row = new Array[n+1]
    
    // Initialize first row
    FOR j FROM 0 TO n:
        previous_row[j] = j
    
    // Process each character of source
    FOR i FROM 1 TO m:
        current_row[0] = i
        
        FOR j FROM 1 TO n:
            IF source[i-1] == target[j-1]:
                cost = 0
            ELSE:
                cost = 1
            
            current_row[j] = MIN(
                current_row[j-1] + 1,      // Insertion
                previous_row[j] + 1,        // Deletion
                previous_row[j-1] + cost    // Substitution
            )
        
        // Swap rows
        SWAP(previous_row, current_row)
    
    RETURN previous_row[n]
```

---

## 📊 Visual Example

Computing distance between `"kitten"` and `"sitting"`:

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3

Distance = 3 (substitute k→s, substitute e→i, insert g)
Similarity = 1 - (3 / 7) = 0.571 or 57.1%
```

---

## 🚀 Usage

### Basic Usage

```rust
use fuzzly::{FuzzySearcher, MatchResult};

fn main() {
    // Create a fuzzy searcher with a threshold (0.0 to 1.0)
    let searcher = FuzzySearcher::new(0.6);
    
    // Define candidates to search through
    let candidates = vec![
        "apple", "application", "applet", 
        "banana", "bandana", "orange"
    ];
    
    // Search for matches
    let results = searcher.search("aple", &candidates);
    
    for result in results {
        println!("{}: {:.1}%", result.text, result.score * 100.0);
    }
}
```

### Output

```
apple: 80.0%
applet: 66.7%
```

### Advanced Usage

```rust
use fuzzly::{FuzzySearcher, LevenshteinDistance};

fn main() {
    // Direct distance calculation
    let distance = LevenshteinDistance::compute("kitten", "sitting");
    println!("Edit distance: {}", distance); // 3
    
    // Get similarity score
    let similarity = LevenshteinDistance::similarity("kitten", "sitting");
    println!("Similarity: {:.1}%", similarity * 100.0); // 57.1%
    
    // Case-insensitive matching
    let searcher = FuzzySearcher::new(0.5)
        .case_insensitive(true);
    
    let results = searcher.search("APPLE", &["apple", "Apple", "APPLE"]);
    // All will match with score 1.0
}
```

---

## 🏗️ Project Structure

```
fuzzy-search-rs/
├── .github/            # Workflows, issue templates, PR template, discussions
├── benches/            # Criterion benchmarks
├── docs/               # GitHub Pages site and branding assets
├── src/
│   ├── lib.rs          # Library: Levenshtein + fuzzy-search implementations
│   └── main.rs         # CLI/demo entry point
├── CHANGELOG.md        # Release notes by version
├── CODE_OF_CONDUCT.md  # Community expectations
├── CONTRIBUTING.md     # Contribution guide
├── Cargo.toml          # Project metadata and dependencies
├── LICENSE             # MIT License
├── README.md           # Project overview and usage guide
└── SECURITY.md         # Security reporting guidance
```

---

## 🔧 Implementation Details

### Time Complexity
- **O(m × n)** where m and n are the lengths of the two strings

### Space Complexity
- **Standard version**: O(m × n) for the full matrix
- **Optimized version**: O(min(m, n)) using two-row technique

### Features
- ✅ Levenshtein distance calculation
- ✅ Similarity scoring (0.0 to 1.0)
- ✅ Threshold-based filtering
- ✅ Case-insensitive matching
- ✅ Memory-optimized implementation
- ✅ Detailed match information (operations performed)
- ✅ Ranked results by relevance

---

## 📖 Implementing in Other Languages

The pseudo code above can be directly translated to any language:

<details>
<summary>Python</summary>

```python
def levenshtein_distance(source: str, target: str) -> int:
    m, n = len(source), len(target)
    
    if m == 0: return n
    if n == 0: return m
    
    # Use two rows for optimization
    prev_row = list(range(n + 1))
    curr_row = [0] * (n + 1)
    
    for i in range(1, m + 1):
        curr_row[0] = i
        for j in range(1, n + 1):
            cost = 0 if source[i-1] == target[j-1] else 1
            curr_row[j] = min(
                curr_row[j-1] + 1,      # insertion
                prev_row[j] + 1,         # deletion
                prev_row[j-1] + cost     # substitution
            )
        prev_row, curr_row = curr_row, prev_row
    
    return prev_row[n]
```
</details>

<details>
<summary>JavaScript</summary>

```javascript
function levenshteinDistance(source, target) {
    const m = source.length;
    const n = target.length;
    
    if (m === 0) return n;
    if (n === 0) return m;
    
    let prevRow = Array.from({ length: n + 1 }, (_, i) => i);
    let currRow = new Array(n + 1).fill(0);
    
    for (let i = 1; i <= m; i++) {
        currRow[0] = i;
        for (let j = 1; j <= n; j++) {
            const cost = source[i-1] === target[j-1] ? 0 : 1;
            currRow[j] = Math.min(
                currRow[j-1] + 1,      // insertion
                prevRow[j] + 1,         // deletion
                prevRow[j-1] + cost     // substitution
            );
        }
        [prevRow, currRow] = [currRow, prevRow];
    }
    
    return prevRow[n];
}
```
</details>

<details>
<summary>Go</summary>

```go
func levenshteinDistance(source, target string) int {
    m, n := len(source), len(target)
    
    if m == 0 { return n }
    if n == 0 { return m }
    
    prevRow := make([]int, n+1)
    currRow := make([]int, n+1)
    
    for j := 0; j <= n; j++ {
        prevRow[j] = j
    }
    
    for i := 1; i <= m; i++ {
        currRow[0] = i
        for j := 1; j <= n; j++ {
            cost := 1
            if source[i-1] == target[j-1] {
                cost = 0
            }
            currRow[j] = min(
                currRow[j-1]+1,      // insertion
                prevRow[j]+1,         // deletion
                prevRow[j-1]+cost,    // substitution
            )
        }
        prevRow, currRow = currRow, prevRow
    }
    
    return prevRow[n]
}
```
</details>

---

## ⚡ Performance

### Running benchmarks

```sh
cargo bench
```

HTML reports with charts are written to `target/criterion/`. Open any
`target/criterion/<group>/report/index.html` in a browser to see the full
analysis.

Quick run (without the slow measurement warmup):

```sh
cargo bench -- --quick
```

### What is benchmarked

The single `levenshtein` benchmark group runs all three implementations on the
same input, at five stress levels, so you can see the progression side-by-side.

| Tier | String lengths | What it stresses |
|------|---------------|-----------------|
| tiny | 6 vs 7 | Function-call overhead dominates — gap is small |
| short | 22 vs 24 | Matrix allocation cost starts to show |
| medium | 66 vs 68 | Cache pressure — full matrix no longer fits in L1 |
| long | 182 vs 191 | O(m×n) full matrix vs two-row gap fully visible |
| stress | 270 vs 268 | Worst-case: alphabet vs reversed alphabet, every DP cell forced |

### Results

Measured on an x86-64 Linux machine (median of 100 samples, release build).
Numbers are the **median** time per call.

| Tier | `compute` | `compute_optimized` | `compute_fast` | `fast` vs `compute` |
|------|----------:|--------------------:|---------------:|:-------------------:|
| tiny (6 vs 7) | 445 ns | 330 ns | **184 ns** | **2.4× faster** |
| short (22 vs 24) | 2,772 ns | 1,374 ns | **1,206 ns** | **2.3× faster** |
| medium (66 vs 68) | 14,755 ns | 10,297 ns | **10,602 ns** | **1.4× faster** |
| long (182 vs 191) | 104,210 ns | 112,580 ns | **87,915 ns** | **1.2× faster** |
| stress (270 vs 268) | 119,920 ns | **82,453 ns** | 95,073 ns | 1.3× faster |

> **How to read the table:**
> - `compute` is the learning-friendly full-matrix baseline.
> - `compute_optimized` saves memory (two rows instead of full matrix) but keeps
>   `usize` (8-byte) cells — it wins on the stress tier where the full matrix
>   exceeds cache capacity.
> - `compute_fast` adds `u32` cells (4 bytes, 2× cache density) and an ASCII
>   byte path (no `Vec<char>` allocation) and wins at every tier except the
>   stress tier where `compute_optimized`'s layout happens to fit the CPU's
>   prefetcher slightly better.
>
> Reproduce on your machine: `cargo bench`

### Optimisation tiers

Three implementations are provided, each building on the previous:

| Method | Time | Space | Notes |
|--------|------|-------|-------|
| `compute` | O(m·n) | O(m·n) | Full matrix — easiest to follow |
| `compute_optimized` | O(m·n) | O(min(m,n)) | Two-row rolling buffer, `usize` cells |
| `compute_fast` | O(m·n) | O(min(m,n)) | Two-row rolling buffer, **`u32` cells** + **ASCII byte path** |

#### Key techniques in `compute_fast`

1. **ASCII fast path** — if both strings are ASCII, `as_bytes()` is used directly
   instead of collecting a `Vec<char>`. This eliminates a heap allocation per call
   and keeps the data as a tight byte slice.

2. **`u32` DP cells** — each row element is 4 bytes instead of 8 (`usize` on
   64-bit). The two rolling rows therefore fit in half as much cache, which
   improves throughput especially for short and medium strings.

3. **Length-difference early exit in `search`** — before running the DP,
   `FuzzySearcher::search` checks whether the length difference between the query
   and a candidate already makes it impossible to reach the similarity threshold.
   Candidates that can't possibly match are skipped without computing anything.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome.

- Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request
- Use [Discussions](https://github.com/Fanaperana/fuzzy-search-rs/discussions) for questions and ideas
- Open issues for bugs or feature requests using the repository templates
- Report sensitive issues through the guidance in [SECURITY.md](SECURITY.md)

---

## 🔗 Related Resources

- [Levenshtein Distance (Wikipedia)](https://en.wikipedia.org/wiki/Levenshtein_distance)
- [Wagner-Fischer Algorithm](https://en.wikipedia.org/wiki/Wagner%E2%80%93Fischer_algorithm)
- [fzf - Command-line fuzzy finder](https://github.com/junegunn/fzf)
