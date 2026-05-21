use crate::model::{LMTransformer, LMTransformerConfig};
use crate::tokenizer::BpeTokenizer;
use burn::{
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{backend::Backend, Int, Tensor},
};
use std::path::PathBuf;

/// Параметры генерации
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_steps: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub max_len: usize,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_steps: 100,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            max_len: 128,
        }
    }
}

/// Основная функция генерации
/// Основная функция генерации
pub fn generate<B: Backend>(device: &B::Device, prompt: &str, config: GenerateConfig) -> String {
    // ── 1. Load tokenizer ────────────────────────────────────────────
    let tokenizer = load_tokenizer();
    let eos = tokenizer.eos_token;

    // ── 2. Load model ────────────────────────────────────────────────
    let model_path = "artifacts/model";
    let model: LMTransformer<B> =
        match CompactRecorder::new().load(PathBuf::from(model_path), device) {
            Ok(record) => {
                println!("Model loaded successfully from {}", model_path);
                let model_config = LMTransformerConfig {
                    d_model: 256,
                    n_heads: 8,
                    n_layers: 6,
                    vocab_size: tokenizer.vocab_size(),
                    max_len: config.max_len,
                };
                model_config.init(device).load_record(record)
            }
            Err(e) => panic!("Failed to load model from {}: {}", model_path, e),
        };

    // ── 3. Encode prompt ─────────────────────────────────────────────
    let mut ids: Vec<usize> = tokenizer.encode(prompt);
    if ids.is_empty() {
        ids.push(tokenizer.bos_token);
    }
    println!("Prompt tokens: {}", ids.len());

    // ── 4. Autoregressive generation ────────────────────────────────
    for step in 0..config.max_steps {
        if ids.len() >= config.max_len {
            break;
        }

        // Подготавливаем input (последние max_len токенов)
        let input_ids: Vec<usize> = if ids.len() > config.max_len {
            ids[ids.len() - config.max_len..].to_vec()
        } else {
            ids.clone()
        };

        // ── ФИКС ЗДЕСЬ ─────────────────────────────────────────────
        let input_data: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();

        let input_tensor = Tensor::<B, 2, Int>::from_data(input_data.as_slice(), device)
            .reshape([1, input_ids.len() as i64]);

        let logits = model.forward(input_tensor);

        // Берём logits последнего токена
        let last_logits = logits
            .slice([
                0..1,
                (input_ids.len() - 1)..input_ids.len(),
                0..tokenizer.vocab_size(),
            ])
            .reshape([tokenizer.vocab_size()]);

        // Sampling
        let next_id = sample_token(last_logits, config.temperature, config.top_k, config.top_p);

        if next_id == eos {
            println!("EOS token generated.");
            break;
        }

        ids.push(next_id);

        if step % 10 == 0 {
            print!(".");
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
    }

    println!();
    tokenizer.decode(&ids)
}

/// Sampling: temperature + top-k + top-p
fn sample_token<B: Backend>(
    logits: Tensor<B, 1>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> usize {
    let logits = logits / temperature;
    let probs = burn::tensor::activation::softmax(logits, 0);

    let probs_data = probs.into_data();
    let probs_slice = probs_data
        .as_slice::<f32>()
        .expect("Failed to read probabilities");

    let mut indices: Vec<usize> = (0..probs_slice.len()).collect();
    indices.sort_by(|&a, &b| probs_slice[b].partial_cmp(&probs_slice[a]).unwrap());

    let mut cumulative = 0.0f32;
    let mut valid_tokens = Vec::new();

    for &idx in indices.iter().take(top_k) {
        cumulative += probs_slice[idx];
        valid_tokens.push(idx);
        if cumulative >= top_p {
            break;
        }
    }

    use rand::prelude::*;
    let mut rng = rand::thread_rng();
    *valid_tokens.choose(&mut rng).unwrap_or(&0)
}

fn load_tokenizer() -> BpeTokenizer {
    let json = std::fs::read_to_string("data/tokenizer.json")
        .expect("Tokenizer not found! Train the model first.");
    serde_json::from_str(&json).expect("Invalid tokenizer format")
}
