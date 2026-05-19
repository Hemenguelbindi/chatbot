use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// BPE tokenizer trained from scratch on a corpus.
#[derive(Serialize, Deserialize)]
pub struct BpeTokenizer {
    /// token string -> id
    vocab: HashMap<String, usize>,
    /// id -> token string
    ivocab: Vec<String>,
    /// merge rules: (left_id, right_id) -> merged_id
    #[serde(
        serialize_with = "serialize_merges",
        deserialize_with = "deserialize_merges"
    )]
    merges: HashMap<(usize, usize), usize>,

    /// Special tokens — public API for downstream use.
    /// Currently unused by the training loop itself,
    /// but exposed for padding, BOS/EOS injection, etc.
    #[allow(dead_code)]
    pub pad_token: usize,
    #[allow(dead_code)]
    #[serde(skip)]
    pub bos_token: usize,
    #[allow(dead_code)]
    #[serde(skip)]
    pub eos_token: usize,
    #[allow(dead_code)]
    #[serde(skip)]
    pub unk_token: usize,
}

fn serialize_merges<S: Serializer>(
    merges: &HashMap<(usize, usize), usize>,
    s: S,
) -> Result<S::Ok, S::Error> {
    let vec: Vec<((usize, usize), usize)> =
        merges.iter().map(|(k, v)| (*k, *v)).collect();
    vec.serialize(s)
}

fn deserialize_merges<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<HashMap<(usize, usize), usize>, D::Error> {
    let vec: Vec<((usize, usize), usize)> = Vec::deserialize(d)?;
    Ok(vec.into_iter().collect())
}

impl BpeTokenizer {
    /// Create a new (untrained) tokenizer with special tokens.
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        let mut ivocab = Vec::new();
        let specials = ["<pad>", "<bos>", "<eos>", "Ġ"];
        for (i, &tok) in specials.iter().enumerate() {
            vocab.insert(tok.to_string(), i);
            ivocab.push(tok.to_string());
        }
        BpeTokenizer {
            vocab,
            ivocab,
            merges: HashMap::new(),
            pad_token: 0,
            bos_token: 1,
            eos_token: 2,
            unk_token: 3,
        }
    }

    /// Train BPE on the given sentences.
    pub fn train<I, S>(&mut self, sentences: I, target_vocab_size: usize)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        // Rebuild vocab from ivocab (in case of deserialization)
        if self.vocab.is_empty() && !self.ivocab.is_empty() {
            for (i, tok) in self.ivocab.iter().enumerate() {
                self.vocab.insert(tok.clone(), i);
            }
        }

        // 1. Count characters
        let mut char_counts: HashMap<String, usize> = HashMap::new();
        let mut sequences: Vec<Vec<String>> = Vec::new();

        for s in sentences {
            let txt = s.as_ref();
            let chars: Vec<String> = txt.chars().map(|c| c.to_string()).collect();
            for ch in &chars {
                *char_counts.entry(ch.clone()).or_insert(0) += 1;
            }
            sequences.push(chars);
        }

        // Add all seen characters to vocab
        for (ch, _) in char_counts {
            if !self.vocab.contains_key(&ch) {
                let idx = self.ivocab.len();
                self.vocab.insert(ch.clone(), idx);
                self.ivocab.push(ch);
            }
        }

        // 2. Iteratively merge the most frequent pair
        let start_vocab = self.vocab.len();
        while self.vocab.len() < target_vocab_size {
            // Progress indicator every 50 merges
            if (self.vocab.len() - start_vocab) % 50 == 0 {
                eprintln!(
                    "  BPE merge {}/{} (vocab size: {})",
                    self.vocab.len() - start_vocab,
                    target_vocab_size - start_vocab,
                    self.vocab.len()
                );
            }
            // Count pairs
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
            for seq in &sequences {
                for w in seq.windows(2) {
                    let pair = (w[0].clone(), w[1].clone());
                    *pair_counts.entry(pair).or_insert(0) += 1;
                }
            }
            if pair_counts.is_empty() {
                break; // nothing more to merge
            }
            // Find max pair
            let ((a, b), _max) = pair_counts
                .into_iter()
                .max_by_key(|&(_, cnt)| cnt)
                .expect("there is at least one pair");
            let merged = format!("{}{}", a, b);

            // Get or create id for merged token
            let merged_id = if let Some(&id) = self.vocab.get(&merged) {
                id
            } else {
                let id = self.ivocab.len();
                self.vocab.insert(merged.clone(), id);
                self.ivocab.push(merged.clone());
                id
            };

            // Get ids of a and b
            let a_id = *self
                .vocab
                .get(&a)
                .expect("left token must be in vocab");
            let b_id = *self
                .vocab
                .get(&b)
                .expect("right token must be in vocab");
            // Record merge rule
            self.merges.insert((a_id, b_id), merged_id);

            // Apply the merge to all sequences
            for seq in &mut sequences {
                let mut i = 0;
                while i < seq.len() - 1 {
                    if seq[i] == a && seq[i + 1] == b {
                        seq.splice(i..i + 2, std::iter::once(merged.clone()));
                        // after splicing, stay at same i to check for overlapping merges
                    } else {
                        i += 1;
                    }
                }
            }
        }
    }

    /// Convert text to token ids.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        // Start with characters; unknown chars map to unk_token
        let unk = self.unk_token;
        let mut ids: Vec<usize> = text
            .chars()
            .map(|ch| ch.to_string())
            .map(|t| self.vocab.get(&t).copied().unwrap_or(unk))
            .collect();

        // Guard against empty input
        if ids.len() <= 1 {
            return ids;
        }

        // Apply merge rules in order; do multiple passes until no change
        let mut changed = true;
        while changed {
            changed = false;
            let mut i = 0;
            while i + 1 < ids.len() {
                let pair = (ids[i], ids[i + 1]);
                if let Some(&merged) = self.merges.get(&pair) {
                    ids.splice(i..i + 2, std::iter::once(merged));
                    changed = true;
                } else {
                    i += 1;
                }
            }
        }
        ids
    }

    /// Convert token ids back to text.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| self.ivocab.get(id))
            .cloned()
            .collect()
    }

    /// Vocabulary size (including specials).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_tokens() {
        let tok = BpeTokenizer::new();
        assert_eq!(tok.vocab_size(), 4);
        assert_eq!(tok.pad_token, 0);
        assert_eq!(tok.bos_token, 1);
        assert_eq!(tok.eos_token, 2);
        assert_eq!(tok.unk_token, 3);
    }

    #[test]
    fn test_train_and_encode() {
        let mut tok = BpeTokenizer::new();
        let data = ["hello world", "hello"];
        tok.train(data.iter(), 50);
        let ids = tok.encode("hello world");
        let back = tok.decode(&ids);
        // Should be able to roundtrip at least approximately
        assert!(!ids.is_empty());
        assert_eq!(back, "hello world");
    }
}
