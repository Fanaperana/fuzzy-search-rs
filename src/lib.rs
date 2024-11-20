use std::collections::HashMap;
use std::path::Path;
use std::fs;

#[derive(Debug)]
pub struct FuzzySearch {
    // Configurable threshold for string similarity
    threshold: f32,
    // Cache for previously computed scores
    score_cache: HashMap<(String, String), f32>,
}

impl FuzzySearch {
    pub fn new(threshold: f32) -> Self {
        FuzzySearch {
            threshold,
            score_cache: HashMap::new(),
        }
    }

    /// Search for pattern in both filename and content
    pub fn search_file<P: AsRef<Path>>(&mut self, pattern: &str, file_path: P) -> Option<SearchResult> {
        let path = file_path.as_ref();
        
        // Check filename first (faster)
        let filename = path.file_name()?.to_str()?;
        let filename_score = self.compute_similarity(pattern, filename);
        
        // If filename matches well enough, no need to check content
        if filename_score >= self.threshold {
            return Some(SearchResult {
                path: path.to_path_buf(),
                filename_score,
                content_score: 0.0,
                matches_found: vec![],
            });
        }

        // Check file content
        if let Ok(content) = fs::read_to_string(path) {
            let content_score = self.search_content(pattern, &content);
            if content_score >= self.threshold {
                let matches = self.find_matching_lines(pattern, &content);
                return Some(SearchResult {
                    path: path.to_path_buf(),
                    filename_score,
                    content_score,
                    matches_found: matches,
                });
            }
        }

        None
    }

    /// Search for pattern in content string
    pub fn search_content(&mut self, pattern: &str, content: &str) -> f32 {
        content
            .lines()
            .map(|line| self.compute_similarity(pattern, line))
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    /// Compute similarity score between two strings using Wagner-Fischer algorithm
    fn compute_similarity(&mut self, s1: &str, s2: &str) -> f32 {
        let key = (s1.to_string(), s2.to_string());
        
        if let Some(&score) = self.score_cache.get(&key) {
            return score;
        }

        let len1 = s1.chars().count();
        let len2 = s2.chars().count();
        
        if len1 == 0 { return 0.0; }
        if len2 == 0 { return 0.0; }

        let s1_lower: Vec<char> = s1.to_lowercase().chars().collect();
        let s2_lower: Vec<char> = s2.to_lowercase().chars().collect();

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        // Initialize first row and column
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        // Fill in the rest of the matrix
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1_lower[i-1] == s2_lower[j-1] { 0 } else { 1 };
                
                matrix[i][j] = (matrix[i-1][j] + 1)  // deletion
                    .min(matrix[i][j-1] + 1)         // insertion
                    .min(matrix[i-1][j-1] + cost);   // substitution
            }
        }

        let distance = matrix[len1][len2] as f32;
        let max_len = len1.max(len2) as f32;
        let score = 1.0 - (distance / max_len);

        self.score_cache.insert(key, score);
        score
    }

    /// Find matching lines in content with some context
    fn find_matching_lines(&mut self, pattern: &str, content: &str) -> Vec<Match> {
        let mut matches = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        for (i, &line) in lines.iter().enumerate() {
            let score = self.compute_similarity(pattern, line);
            if score >= self.threshold {
                let context = self.get_context(&lines, i);
                matches.push(Match {
                    line_number: i + 1,
                    line: line.to_string(),
                    score,
                    context,
                });
            }
        }

        matches
    }

    /// Get context lines around a match
    fn get_context(&self, lines: &[&str], line_num: usize) -> Vec<String> {
        let context_size = 2; // Number of lines before and after
        let start = line_num.saturating_sub(context_size);
        let end = (line_num + context_size + 1).min(lines.len());

        lines[start..end].iter().map(|&s| s.to_string()).collect()
    }
}

#[derive(Debug)]
pub struct SearchResult {
    pub path: std::path::PathBuf,
    pub filename_score: f32,
    pub content_score: f32,
    pub matches_found: Vec<Match>,
}

#[derive(Debug)]
pub struct Match {
    pub line_number: usize,
    pub line: String,
    pub score: f32,
    pub context: Vec<String>,
}

// // Example usage
// fn main() {
//     let mut fuzzy = FuzzySearch::new(0.7);
    
//     // Search in a specific file
//     if let Some(result) = fuzzy.search_file(".", ".") {
//         println!("Found matches in {:?}", result.path);
//         for matched in result.matches_found {
//             println!("Match on line {}: {}", matched.line_number, matched.line);
//             println!("Context:");
//             for context_line in matched.context {
//                 println!("  {}", context_line);
//             }
//         }
//     }
// }