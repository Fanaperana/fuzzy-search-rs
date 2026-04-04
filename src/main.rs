//! # Fuzzy Search Example
//!
//! This example demonstrates how to use the fuzzly fuzzy search library.
//! Run with: `cargo run`

use fuzzly::{FuzzySearcher, LevenshteinDistance, LevenshteinWithOperations};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Fuzzy Search Algorithm - Demo & Tutorial           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ========================================================================
    // Example 1: Basic Levenshtein Distance
    // ========================================================================
    section("1. Basic Levenshtein Distance");

    let examples = [
        ("kitten", "sitting"),
        ("apple", "aple"),
        ("algorithm", "altruistic"),
        ("hello", "hello"),
        ("cat", "cart"),
    ];

    println!("The Levenshtein distance is the minimum number of single-character");
    println!("edits (insertions, deletions, substitutions) to transform one string");
    println!("into another.\n");

    for (s1, s2) in examples {
        let distance = LevenshteinDistance::compute(s1, s2);
        let similarity = LevenshteinDistance::similarity(s1, s2);
        println!(
            "  '{}' → '{}': distance = {}, similarity = {:.1}%",
            s1,
            s2,
            distance,
            similarity * 100.0
        );
    }

    // ========================================================================
    // Example 2: Detailed Edit Operations
    // ========================================================================
    section("2. Detailed Edit Operations");

    println!("Understanding what operations are needed to transform strings:\n");

    let result = LevenshteinWithOperations::compute("kitten", "sitting");
    println!("{}", result);

    let result = LevenshteinWithOperations::compute("saturday", "sunday");
    println!("{}", result);

    // ========================================================================
    // Example 3: Fuzzy Search Through Candidates
    // ========================================================================
    section("3. Fuzzy Search Through Candidates");

    let candidates = vec![
        "apple",
        "application",
        "applet",
        "appreciate",
        "banana",
        "bandana",
        "orange",
        "grape",
        "grapefruit",
    ];

    println!("Candidates: {:?}\n", candidates);

    // Search with different queries
    let queries = ["aple", "app", "banan", "orang"];

    for query in queries {
        println!("Searching for: '{}' (threshold: 60%)", query);

        let searcher = FuzzySearcher::new(0.6);
        let results = searcher.search(query, &candidates);

        if results.is_empty() {
            println!("  No matches found.\n");
        } else {
            for result in results {
                println!(
                    "  ✓ {} ({:.1}% - {})",
                    result.text,
                    result.score * 100.0,
                    result.quality()
                );
            }
            println!();
        }
    }

    // ========================================================================
    // Example 4: Configuration Options
    // ========================================================================
    section("4. Configuration Options");

    // Case-insensitive matching (default)
    println!("Case-insensitive matching (default):");
    let searcher = FuzzySearcher::new(0.8).case_insensitive(true);
    let results = searcher.search("APPLE", &["apple", "Apple", "APPLE"]);
    for result in &results {
        println!(
            "  '{}' matches with {:.1}%",
            result.text,
            result.score * 100.0
        );
    }
    println!();

    // Case-sensitive matching
    println!("Case-sensitive matching:");
    let searcher = FuzzySearcher::new(0.8).case_insensitive(false);
    let results = searcher.search("Apple", &["apple", "Apple", "APPLE"]);
    for result in &results {
        println!(
            "  '{}' matches with {:.1}%",
            result.text,
            result.score * 100.0
        );
    }
    println!();

    // Limited results
    println!("Limited to top 3 results:");
    let searcher = FuzzySearcher::new(0.3).max_results(3);
    let results = searcher.search("app", &candidates);
    for result in &results {
        println!("  {} ({:.1}%)", result.text, result.score * 100.0);
    }

    // ========================================================================
    // Example 5: Finding the Best Match
    // ========================================================================
    section("5. Finding the Best Match");

    let searcher = FuzzySearcher::new(0.5);
    let query = "progam";

    let programming_terms = vec![
        "program",
        "programmer",
        "programming",
        "progress",
        "diagram",
    ];

    println!("Query: '{}'\n", query);
    println!("Candidates: {:?}\n", programming_terms);

    if let Some(best) = searcher.find_best(query, &programming_terms) {
        println!(
            "Best match: '{}' with {:.1}% similarity ({})",
            best.text,
            best.score * 100.0,
            best.quality()
        );
    }

    // ========================================================================
    // Example 6: Memory-Optimized Version
    // ========================================================================
    section("6. Memory-Optimized Version");

    println!("For very long strings, use the optimized version:");
    println!("(Uses O(min(m,n)) space instead of O(m×n))\n");

    let long_s1 = "The quick brown fox jumps over the lazy dog";
    let long_s2 = "The quick brown cat jumps over the lazy dog";

    let distance_standard = LevenshteinDistance::compute(long_s1, long_s2);
    let distance_optimized = LevenshteinDistance::compute_optimized(long_s1, long_s2);

    println!("String 1: \"{}\"", long_s1);
    println!("String 2: \"{}\"", long_s2);
    println!();
    println!("Standard algorithm: distance = {}", distance_standard);
    println!("Optimized algorithm: distance = {}", distance_optimized);
    println!("Results match: {}", distance_standard == distance_optimized);

    // ========================================================================
    // Example 7: Unicode Support
    // ========================================================================
    section("7. Unicode Support");

    println!("The algorithm fully supports Unicode characters:\n");

    let unicode_examples = [
        ("café", "cafe"),
        ("日本", "日本語"),
        ("🎉🎊", "🎉🎁"),
        ("naïve", "naive"),
    ];

    for (s1, s2) in unicode_examples {
        let distance = LevenshteinDistance::compute(s1, s2);
        let similarity = LevenshteinDistance::similarity(s1, s2);
        println!(
            "  '{}' → '{}': distance = {}, similarity = {:.1}%",
            s1,
            s2,
            distance,
            similarity * 100.0
        );
    }

    // ========================================================================
    // Example 8: Practical Use Case - Command Suggestion
    // ========================================================================
    section("8. Practical Use Case - Command Suggestion");

    let commands = vec![
        "git commit",
        "git push",
        "git pull",
        "git status",
        "git branch",
        "git checkout",
        "git merge",
        "git rebase",
        "git log",
        "git diff",
    ];

    let typos = ["git comit", "git pus", "git chekout", "git stauts"];

    println!("Available commands: {:?}\n", commands);

    for typo in typos {
        let searcher = FuzzySearcher::new(0.7);
        if let Some(best) = searcher.find_best(typo, &commands) {
            println!(
                "  Typed: '{}' → Did you mean: '{}' ({:.0}% match)?",
                typo,
                best.text,
                best.score * 100.0
            );
        }
    }

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    End of Demo                               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

/// Helper function to print section headers
fn section(title: &str) {
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  {}", title);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
}
