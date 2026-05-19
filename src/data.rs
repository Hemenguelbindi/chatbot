use std::sync::Arc;

use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{Int, Tensor, backend::Backend},
};

use crate::tokenizer::BpeTokenizer;

/// A single training example: input IDs and target IDs (shifted by one).
#[derive(Debug, Clone)]
pub struct LMExample {
    pub input_ids: Vec<u64>,
    pub target_ids: Vec<u64>,
}

/// Dataset that yields LMExample.
#[derive(Debug, Clone)]
pub struct LMData {
    pub examples: Vec<LMExample>,
}

impl LMData {
    /// Build dataset from a text string.
    pub fn from_text(text: &str, tokenizer: &BpeTokenizer, max_len: usize) -> Self {
        eprintln!("  Tokenizing text ({} bytes)...", text.len());
        let token_ids: Vec<u64> = tokenizer
            .encode(text)
            .into_iter()
            .map(|x| x as u64)
            .collect();
        eprintln!("  Tokenized into {} tokens, chunking...", token_ids.len());
        let mut examples = Vec::new();
        for chunk in token_ids.chunks(max_len) {
            let chunk: &[u64] = chunk;
            if chunk.len() < 2 {
                continue;
            }
            let mut input = vec![0u64; max_len];
            let mut target = vec![0u64; max_len];
            let len = chunk.len();
            input[..len].copy_from_slice(chunk);
            // target is input shifted left by one; last position is padding (ignored)
            target[..len - 1].copy_from_slice(&chunk[1..]);
            examples.push(LMExample {
                input_ids: input,
                target_ids: target,
            });
        }
        Self { examples }
    }

    /// Split into train/test.
    pub fn split(self, ratio: f32) -> (Self, Self) {
        let split_idx = (self.examples.len() as f32 * ratio) as usize;
        let (train, test) = self.examples.split_at(split_idx);
        (
            Self {
                examples: train.to_vec(),
            },
            Self {
                examples: test.to_vec(),
            },
        )
    }
}

impl Dataset<LMExample> for LMData {
    fn get(&self, index: usize) -> Option<LMExample> {
        self.examples.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.examples.len()
    }
}

/// Batch struct for Burn.
#[derive(Debug, Clone)]
pub struct LMBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,  // [batch, seq_len]
    pub targets: Tensor<B, 2, Int>, // [batch, seq_len]
}

/// Batcher turns Vec<LMExample> into tensors.
#[derive(Clone)]
pub struct LMBatcher {
    #[allow(dead_code)]
    tokenizer: Arc<BpeTokenizer>,
    max_len: usize,
}

impl LMBatcher {
    pub fn new(tokenizer: Arc<BpeTokenizer>, max_len: usize) -> Self {
        Self { tokenizer, max_len }
    }
}

impl<B: Backend> Batcher<B, LMExample, LMBatch<B>> for LMBatcher {
    fn batch(&self, items: Vec<LMExample>, device: &B::Device) -> LMBatch<B> {
        let batch_size = items.len();
        let mut input_vec = Vec::with_capacity(batch_size * self.max_len);
        let mut target_vec = Vec::with_capacity(batch_size * self.max_len);

        for item in &items {
            input_vec.extend(item.input_ids.iter().map(|&x| x as i64));
            target_vec.extend(item.target_ids.iter().map(|&x| x as i64));
        }

        let inputs = Tensor::<B, 1, Int>::from_data(input_vec.as_slice(), device)
            .reshape([batch_size, self.max_len]);
        let targets = Tensor::<B, 1, Int>::from_data(target_vec.as_slice(), device)
            .reshape([batch_size, self.max_len]);

        LMBatch { inputs, targets }
    }
}
