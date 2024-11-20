use std::cmp::{min, Ordering};

#[derive(Debug)]
pub struct FuzzyMatch {
    text: String,
    score: f32,
    ratio: f32,
    partial_ratio: f32,
    operations: Vec<String>,
}

impl FuzzyMatch {
    // Helper method to create a new FuzzyMatch
    fn new(text: String, score: f32, ratio: f32, partial_ratio: f32, operations: Vec<String>) -> Self {
        FuzzyMatch {
            text,
            score,
            ratio,
            partial_ratio,
            operations,
        }
    }
    
    // Helper method to get match quality description
    pub fn get_match_quality(&self) -> &str {
        if self.ratio > 95.0 {
            "Excellent"
        } else if self.ratio > 85.0 {
            "Very Good"
        } else if self.ratio > 75.0 {
            "Good"
        } else if self.ratio > 65.0 {
            "Fair"
        } else {
            "Poor"
        }
    }
}

#[derive(Debug)]
pub enum FuzzyError {
    EmptyString,
    InvalidLength,
    ProcessingError(String),
}

pub struct FuzzyMatcher {
    strings: Vec<String>,
}

impl FuzzyMatcher {
    pub fn new(strings: Vec<String>) -> Result<Self, FuzzyError> {
        // Validate input strings
        if strings.is_empty() {
            return Err(FuzzyError::EmptyString);
        }
        
        // Check for empty strings in the vector
        if strings.iter().any(|s| s.is_empty()) {
            return Err(FuzzyError::InvalidLength);
        }
        
        println!("Creating FuzzyMatcher with {} strings", strings.len());
        Ok(FuzzyMatcher { strings })
    }

    fn ratio(&self, s1: &str, s2: &str) -> Result<f32, FuzzyError> {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();
        
        match (len1, len2) {
            (0, 0) => Ok(100.0),
            (0, _) | (_, 0) => Ok(0.0),
            (_, _) => {
                let distance = self.levenshtein_distance(s1, s2)?;
                let length_sum = len1 + len2;
                Ok(((length_sum as f32 - distance as f32) / length_sum as f32) * 100.0)
            }
        }
    }

    fn partial_ratio(&self, s1: &str, s2: &str) -> Result<f32, FuzzyError> {
        let (shorter, longer) = if s1.len() < s2.len() {
            (s1, s2)
        } else {
            (s2, s1)
        };
    
        let shorter_len = shorter.chars().count();
        if shorter_len == 0 {
            return Err(FuzzyError::EmptyString);
        }
    
        let mut best_ratio: f32 = 0.0;
        for i in 0..=longer.len() - shorter.len() {
            let substring = &longer[i..i + shorter.len()];
            let current_ratio = self.ratio(shorter, substring)?;
            best_ratio = f32::max(best_ratio, current_ratio);
        }
    
        Ok(best_ratio)
    }

    fn levenshtein_distance(&self, s1: &str, s2: &str) -> Result<usize, FuzzyError> {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();
        
        match (len1, len2) {
            (0, l2) => Ok(l2),
            (l1, 0) => Ok(l1),
            (_l1, _l2) => {
                let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

                for i in 0..=len1 {
                    matrix[i][0] = i;
                }
                for j in 0..=len2 {
                    matrix[0][j] = j;
                }

                for (i, c1) in s1.chars().enumerate() {
                    for (j, c2) in s2.chars().enumerate() {
                        let substitution_cost = if c1 == c2 { 0 } else { 1 };
                        matrix[i + 1][j + 1] = min(
                            matrix[i][j + 1] + 1,
                            min(
                                matrix[i + 1][j] + 1,
                                matrix[i][j] + substitution_cost
                            )
                        );
                    }
                }

                Ok(matrix[len1][len2])
            }
        }
    }

    // Fast version that only returns essential scores
    fn calculate_score_fast(&self, s1: &str, s2: &str) -> Result<(f32, f32, f32), FuzzyError> {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();

        if len1 == 0 || len2 == 0 {
            return Err(FuzzyError::EmptyString);
        }

        let mut prev_row = vec![0; len2 + 1];
        let mut curr_row = vec![0; len2 + 1];

        // Initialize first row
        for j in 0..=len2 {
            prev_row[j] = j;
        }

        // Calculate edit distance using only two rows
        for (i, c1) in s1.chars().enumerate() {
            curr_row[0] = i + 1;

            for (j, c2) in s2.chars().enumerate() {
                let substitution_cost = if c1 == c2 { 0 } else { 1 };
                curr_row[j + 1] = min(
                    curr_row[j] + 1,
                    min(
                        prev_row[j + 1] + 1,
                        prev_row[j] + substitution_cost
                    )
                );
            }
            
            // Swap rows
            std::mem::swap(&mut prev_row, &mut curr_row);
        }

        // Calculate scores
        let distance = prev_row[len2];
        let score = distance as f32 / len1.max(len2) as f32;
        let ratio = ((len1 + len2) as f32 - distance as f32) / (len1 + len2) as f32 * 100.0;
        
        // Calculate partial ratio using optimized sliding window
        let mut best_partial = 0.0;
        if len1 != len2 {
            let (shorter, longer) = if len1 < len2 { (s1, s2) } else { (s2, s1) };
            let shorter_len = shorter.len();
            
            for i in 0..=longer.len() - shorter_len {
                let substring = &longer[i..i + shorter_len];
                let window_ratio = self.ratio(shorter, substring)?;
                best_partial = f32::max(best_partial, window_ratio);
            }
        } else {
            best_partial = ratio;
        }

        Ok((score, ratio, best_partial))
    }

    // Helper method to find matches using the fast calculator
    pub fn find_matches_fast(&self, query: &str, threshold: f32) -> Result<Vec<(String, f32, f32, f32)>, FuzzyError> {
        if query.is_empty() {
            return Err(FuzzyError::EmptyString);
        }

        let mut matches = Vec::new();
        for s in &self.strings {
            if let Ok((score, ratio, partial_ratio)) = self.calculate_score_fast(query, s) {
                if score <= threshold {
                    matches.push((s.clone(), score, ratio, partial_ratio));
                }
            }
        }

        matches.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
        });

        Ok(matches)
    }

    fn calculate_score(&self, s1: &str, s2: &str) -> Result<(f32, Vec<String>, f32, f32), FuzzyError> {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();

        if len1 == 0 || len2 == 0 {
            return Err(FuzzyError::EmptyString);
        }

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];
        let mut ops = vec![vec![Vec::new(); len2 + 1]; len1 + 1];

        // Initialize matrices...
        for i in 0..=len1 {
            matrix[i][0] = i;
            if i > 0 {
                ops[i][0] = vec![format!("delete {}", s1.chars().nth(i-1).unwrap())];
            }
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
            if j > 0 {
                ops[0][j] = vec![format!("insert {}", s2.chars().nth(j-1).unwrap())];
            }
        }

        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        // Calculate edit distance and operations...
        for (i, c1) in s1_chars.iter().enumerate() {
            for (j, c2) in s2_chars.iter().enumerate() {
                let substitution_cost = if c1 == c2 { 0 } else { 1 };
                
                let delete_cost = matrix[i][j + 1] + 1;
                let insert_cost = matrix[i + 1][j] + 1;
                let sub_cost = matrix[i][j] + substitution_cost;

                matrix[i + 1][j + 1] = min(delete_cost, min(insert_cost, sub_cost));

                let current_ops = if sub_cost == matrix[i + 1][j + 1] {
                    let mut ops = ops[i][j].clone();
                    if c1 != c2 {
                        ops.push(format!("replace {} → {}", c1, c2));
                    }
                    ops
                } else if delete_cost == matrix[i + 1][j + 1] {
                    let mut ops = ops[i][j + 1].clone();
                    ops.push(format!("delete {}", c1));
                    ops
                } else {
                    let mut ops = ops[i + 1][j].clone();
                    ops.push(format!("insert {}", c2));
                    ops
                };

                ops[i + 1][j + 1] = current_ops;
            }
        }

        let score = matrix[len1][len2] as f32 / len1.max(len2) as f32;
        let final_operations = ops[len1][len2].clone();
        
        // Calculate ratios
        let ratio = self.ratio(s1, s2)?;
        let partial_ratio = self.partial_ratio(s1, s2)?;
        
        println!("Comparing '{}' vs '{}':", s1, s2);
        println!("  - Operations needed: {:?}", final_operations);
        println!("  - Score: {:.2} ({} operations / max length {})", 
                score, matrix[len1][len2], len1.max(len2));
        println!("  - Ratio: {:.1}%", ratio);
        println!("  - Partial Ratio: {:.1}%", partial_ratio);
        
        Ok((score, final_operations, ratio, partial_ratio))
    }

    pub fn find_matches(&self, query: &str, threshold: f32) -> Result<Vec<FuzzyMatch>, FuzzyError> {
        if query.is_empty() {
            return Err(FuzzyError::EmptyString);
        }

        println!("\nSearching for matches to '{}' with threshold {}", query, threshold);
        
        let mut matches = Vec::new();
        for s in &self.strings {
            match self.calculate_score(query, s) {
                Ok((score, operations, ratio, partial_ratio)) if score <= threshold => {
                    matches.push(FuzzyMatch::new(
                        s.clone(),
                        score,
                        ratio,
                        partial_ratio,
                        operations
                    ));
                }
                Ok(_) => continue,
                Err(e) => println!("Error processing '{}': {:?}", s, e),
            }
        }

        matches.sort_by(|a, b| {
            a.score.partial_cmp(&b.score)
                .unwrap_or(Ordering::Equal)
        });

        println!("\nFound {} matches", matches.len());
        Ok(matches)
    }
}

fn main() -> Result<(), FuzzyError> {
    let strings = vec![
        "apple".to_string(),
        "appl".to_string(),
        "applet".to_string(),
        "application".to_string(),
        "banana".to_string(),
    ];

    let matcher = FuzzyMatcher::new(strings)?;
    let matches = matcher.find_matches_fast("aple", 0.5)?;
    
    println!("\nResults:");
    println!("-----------------");
    for (text, score, ratio, partial_ratio) in matches {
        println!("Match: {} (score: {:.2}, ratio: {:.1}%, partial_ratio: {:.1}%)", 
                text, score, ratio, partial_ratio);
    }
    
    Ok(())
}