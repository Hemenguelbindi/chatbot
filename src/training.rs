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
        metric::{LossMetric, PerplexityMetric},
        Learner, SequenceOutput, SupervisedTraining, TrainOutput, TrainStep,
    },
};

use crate::data::{LMBatch, LMBatcher, LMData};
use crate::model::{LMTransformer, LMTransformerConfig};
use crate::tokenizer::BpeTokenizer;

/// Максимальное количество строк для загрузки
const MAX_DATASET_LINES: usize = 999_999; // практически без ограничения

// ── TrainStep ────────────────────────────────────────────────────────────

impl<B: AutodiffBackend> TrainStep for LMTransformer<B> {
    type Input = LMBatch<B>;
    type Output = SequenceOutput<B>;

    fn step(&self, item: Self::Input) -> TrainOutput<Self::Output> {
        let logits = self.forward(item.inputs);
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
        let device = logits.device();

        let [b, t, v] = logits.dims();
        let flat_logits = logits.clone().reshape([b * t, v]);
        let [b2, t2] = item.targets.dims();
        let flat_targets = item.targets.clone().reshape([b2 * t2]);

        let loss_fn = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![0]))
            .init(&device);

        let loss = loss_fn.forward(flat_logits, flat_targets);
        SequenceOutput::new(loss, logits, None, item.targets)
    }
}

// ── Training entry point ─────────────────────────────────────────────────

pub fn run<B: AutodiffBackend>(device: B::Device, data_path: Option<&str>) {
    // ── 1. Tokenizer ────────────────────────────────────────────────
    let data_file = data_path.unwrap_or("data/train.txt");
    let tokenizer_path = "data/tokenizer.json";
    let raw = std::fs::read_to_string(data_file)
        .unwrap_or_else(|e| panic!("Файл не найден: {data_file}\n{e}"));

    let tokenizer = if std::path::Path::new(tokenizer_path).exists() {
        eprintln!("Loading cached tokenizer from {tokenizer_path}...");
        let json = std::fs::read_to_string(tokenizer_path).expect("read tokenizer");
        serde_json::from_str(&json).expect("parse tokenizer")
    } else {
        eprintln!("Training new tokenizer on data...");
        let mut tok = BpeTokenizer::new();
        // Обучаем токенизатор на большем объёме
        let subset: String = raw.lines().take(2000).collect::<Vec<_>>().join("\n");
        tok.train(std::iter::once(subset.as_str()), 12000); // ← Увеличен vocab
        std::fs::create_dir_all("data").ok();
        std::fs::write(tokenizer_path, serde_json::to_string(&tok).unwrap())
            .expect("save tokenizer");
        tok
    };

    let pad_token = tokenizer.pad_token;
    let tokenizer = Arc::new(tokenizer);

    // ── 2. Dataset ──────────────────────────────────────────────────
    let dataset_text: String = raw
        .lines()
        .take(MAX_DATASET_LINES)
        .collect::<Vec<_>>()
        .join("\n");
    eprintln!(
        "Building dataset from {} lines ({} bytes)...",
        dataset_text.lines().count(),
        dataset_text.len()
    );

    let data = LMData::from_text(&dataset_text, &tokenizer, 128); // ← max_len = 128
    let (train_data, test_data) = data.split(0.9);

    println!(
        "Dataset: {} train / {} test examples",
        train_data.examples.len(),
        test_data.examples.len()
    );

    // ── 3. Model Config ─────────────────────────────────────────────
    let config = LMTransformerConfig {
        d_model: 256, // увеличили
        n_heads: 8,
        n_layers: 6, // увеличили
        vocab_size: tokenizer.vocab_size(),
        max_len: 128, // ← важно!
    };

    let model: LMTransformer<B> = config.init(&device);
    eprintln!(
        "New model initialized: d_model={}, layers={}, heads={}, vocab={}",
        config.d_model, config.n_layers, config.n_heads, config.vocab_size
    );

    // ── 4. DataLoaders ──────────────────────────────────────────────
    let batcher = LMBatcher::new(tokenizer.clone(), 128);
    let train_dl = DataLoaderBuilder::new(batcher.clone())
        .batch_size(32) // можно попробовать 64, если хватает памяти
        .num_workers(2)
        .build(train_data);

    let test_dl = DataLoaderBuilder::new(batcher)
        .batch_size(32)
        .num_workers(2)
        .build(test_data);

    // ── 5. Optimizer + Scheduler ────────────────────────────────────
    let optim = AdamConfig::new().init();

    let lr_scheduler = burn::lr_scheduler::noam::NoamLrSchedulerConfig::new(5e-4)
        .with_warmup_steps(400)
        .with_model_size(256)
        .init()
        .unwrap();

    // ── 6. Trainer ──────────────────────────────────────────────────
    let trainer = SupervisedTraining::new("artifacts", train_dl, test_dl)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(PerplexityMetric::new().with_pad_token(pad_token))
        .metric_valid_numeric(PerplexityMetric::new().with_pad_token(pad_token))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(5) // на первом этапе 5–15 эпох обычно достаточно
        .summary();

    let learner = Learner::new(model, optim, lr_scheduler);
    let result = trainer.launch(learner);

    // ── 7. Save model ───────────────────────────────────────────────
    let model_path = "artifacts/model";
    CompactRecorder::new()
        .record(result.model.into_record(), PathBuf::from(model_path))
        .expect("save model");

    eprintln!("Обучение завершено. Модель сохранена в {model_path}");
}
