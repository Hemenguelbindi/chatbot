use std::path::PathBuf;
use std::sync::Arc;

use burn::{
    data::dataloader::DataLoaderBuilder,
    nn::loss::CrossEntropyLossConfig,
    optim::AdamConfig,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        Learner, SequenceOutput, SupervisedTraining, TrainOutput, TrainStep,
        metric::{LossMetric, PerplexityMetric},
    },
};
use serde::{Deserialize, Serialize};

use crate::data::{LMBatch, LMBatcher, LMData};
use crate::model::{LMTransformer, LMTransformerConfig};
use crate::tokenizer::BpeTokenizer;

// ── Checkpoint metadata ──────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    epochs_done: usize,
    total_epochs: usize,
    best_perplexity: f64,
}

// ── TrainStep ────────────────────────────────────────────────────────────

impl<B: AutodiffBackend> TrainStep for LMTransformer<B> {
    type Input = LMBatch<B>;
    type Output = SequenceOutput<B>;

    fn step(&self, item: Self::Input) -> TrainOutput<Self::Output> {
        let logits = self.forward(item.inputs); // [batch, seq_len, vocab]
        let device = logits.device();

        let [b, t, v] = logits.dims();
        let flat_logits = logits.clone().reshape([b * t, v]);
        let [b2, t2] = item.targets.dims();
        let flat_targets = item.targets.clone().reshape([b2 * t2]);

        let loss_fn = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![0]))
            .init(&device);
        let loss = loss_fn.forward(flat_logits, flat_targets);

        let output = SequenceOutput::new(loss, logits, None, item.targets);
        let grads = output.loss.backward();
        TrainOutput::new(self, grads, output)
    }
}

// ── InferenceStep ────────────────────────────────────────────────────────

impl<B: Backend> burn::train::InferenceStep for LMTransformer<B> {
    type Input = LMBatch<B>;
    type Output = SequenceOutput<B>;

    fn step(&self, item: Self::Input) -> Self::Output {
        let logits = self.forward(item.inputs);
        SequenceOutput::new(
            Tensor::zeros([1], &logits.device()),
            logits,
            None,
            item.targets,
        )
    }
}

// ── Training entry point ─────────────────────────────────────────────────

pub fn run<B: AutodiffBackend>(device: B::Device) {
    // --------------------------------------------------------------
    // 1. Prepare tokenizer (train on corpus if not cached)
    // --------------------------------------------------------------
    let tokenizer_path = "data/tokenizer.json";
    let raw = std::fs::read_to_string("data/train.txt").expect("need data/train.txt");

    let tokenizer = if std::path::Path::new(tokenizer_path).exists() {
        eprintln!("Loading cached tokenizer from {tokenizer_path}...");
        let json = std::fs::read_to_string(tokenizer_path).expect("read tokenizer");
        serde_json::from_str(&json).expect("parse tokenizer")
    } else {
        eprintln!("Training tokenizer on data subset...");
        let mut tok = BpeTokenizer::new();
        let subset: String = raw.lines().take(500).collect::<Vec<_>>().join("\n");
        tok.train(std::iter::once(subset.as_str()), 2000);
        std::fs::create_dir_all("data").ok();
        std::fs::write(
            tokenizer_path,
            serde_json::to_string(&tok).expect("serialize tokenizer"),
        )
        .expect("write tokenizer");
        eprintln!("Tokenizer saved to {tokenizer_path}");
        tok
    };
    let pad_token = tokenizer.pad_token;
    let tokenizer = Arc::new(tokenizer);

    // --------------------------------------------------------------
    // 2. Build dataset
    // --------------------------------------------------------------
    let dataset_text: String = raw.lines().take(10000).collect::<Vec<_>>().join("\n");
    eprintln!(
        "Building dataset from {} lines ({} bytes)...",
        dataset_text.lines().count(),
        dataset_text.len()
    );
    let data = LMData::from_text(&dataset_text, &tokenizer, 32);
    drop(raw);
    drop(dataset_text);
    let sample_ids: Option<Vec<usize>> = data
        .examples
        .first()
        .map(|ex| ex.input_ids.iter().take(10).map(|&x| x as usize).collect());
    let (train_data, test_data) = data.split(0.9);
    println!(
        "Dataset: {} train / {} test examples",
        train_data.examples.len(),
        test_data.examples.len()
    );

    // --------------------------------------------------------------
    // 3. DataLoaders
    // --------------------------------------------------------------
    let batcher = LMBatcher::new(tokenizer.clone(), 32);
    let train_dl = DataLoaderBuilder::new(batcher.clone())
        .batch_size(16)
        .num_workers(1)
        .build(train_data);
    let test_dl = DataLoaderBuilder::new(batcher)
        .batch_size(16)
        .num_workers(1)
        .build(test_data);

    // --------------------------------------------------------------
    // 4. Model — init or resume from checkpoint
    // --------------------------------------------------------------
    let num_epochs: usize = 5;
    let model_path = "artifacts/model";
    let ckpt_path = "artifacts/checkpoint.json";

    let (model, epochs_done) = load_or_init::<B>(&device, &tokenizer, model_path, ckpt_path);
    let remaining = num_epochs.saturating_sub(epochs_done);

    eprintln!(
        "Model: d_model=32, n_heads=2, n_layers=1, vocab={}",
        tokenizer.vocab_size()
    );
    if epochs_done > 0 {
        eprintln!(
            "Resuming from checkpoint: {}/{} epochs done, {} remaining",
            epochs_done, num_epochs, remaining
        );
    } else {
        eprintln!("Starting fresh training: {} epochs", num_epochs);
    }

    if remaining == 0 {
        println!("All {num_epochs} epochs already completed.");
        return;
    }

    // --------------------------------------------------------------
    // 5. Optimizer, scheduler (fresh each run — Adam rebuilds momentum)
    // --------------------------------------------------------------
    let optim = AdamConfig::new().init();
    let lr_scheduler = burn::lr_scheduler::noam::NoamLrSchedulerConfig::new(5e-4)
        .with_warmup_steps(10)
        .with_model_size(32)
        .init()
        .unwrap();

    // --------------------------------------------------------------
    // 6. Trainer
    // --------------------------------------------------------------
    let trainer = SupervisedTraining::new("artifacts", train_dl, test_dl)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(PerplexityMetric::new().with_pad_token(pad_token))
        .metric_valid_numeric(PerplexityMetric::new().with_pad_token(pad_token))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(remaining)
        .summary();

    let result = trainer.launch(Learner::new(model, optim, lr_scheduler));

    // --------------------------------------------------------------
    // 7. Save model + checkpoint metadata
    // --------------------------------------------------------------
    CompactRecorder::new()
        .record(result.model.into_record(), PathBuf::from(model_path))
        .expect("save model");

    let total_done = epochs_done + remaining;
    save_checkpoint(ckpt_path, total_done, num_epochs, f64::MAX);
    println!(
        "Checkpoint saved: {}/{} epochs — artifacts/",
        total_done, num_epochs
    );

    if let Some(ref ids) = sample_ids {
        println!("Sample decode: {:?}", tokenizer.decode(ids));
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn load_or_init<B: Backend>(
    device: &B::Device,
    tokenizer: &BpeTokenizer,
    model_path: &str,
    ckpt_path: &str,
) -> (LMTransformer<B>, usize) {
    let config = LMTransformerConfig {
        d_model: 32,
        n_heads: 2,
        n_layers: 1,
        vocab_size: tokenizer.vocab_size(),
        max_len: 32,
    };

    // Check for existing checkpoint
    let ckpt = std::fs::read_to_string(ckpt_path)
        .ok()
        .and_then(|s| serde_json::from_str::<Checkpoint>(&s).ok());

    let model_path_buf = PathBuf::from(model_path);

    match ckpt {
        Some(ckpt) => {
            eprintln!(
                "Loading model from {model_path} ({} epochs done)...",
                ckpt.epochs_done
            );
            match CompactRecorder::new().load(model_path_buf.clone(), device) {
                Ok(record) => {
                    let model = config.init(device).load_record(record);
                    (model, ckpt.epochs_done)
                }
                Err(e) => {
                    eprintln!("Failed to load checkpoint: {e}, starting fresh");
                    (config.init(device), 0)
                }
            }
        }
        _ => {
            let model = config.init(device);
            (model, 0)
        }
    }
}

fn save_checkpoint(path: &str, epochs_done: usize, total_epochs: usize, best_perplexity: f64) {
    let ckpt = Checkpoint {
        epochs_done,
        total_epochs,
        best_perplexity,
    };
    std::fs::create_dir_all("artifacts").ok();
    std::fs::write(path, serde_json::to_string(&ckpt).unwrap())
        .expect("write checkpoint");
}
