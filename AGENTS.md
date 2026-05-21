# Chatbot — Language Model Training Project

## Overview

Decoder-only transformer language model trained from scratch on Russian dialog data.
Built with Burn 0.21. Educational project — understanding full LM training pipeline:
data → BPE tokenizer → dataset → transformer → training → perplexity evaluation.

## Architecture

```
Input tokens [batch, 32]
  → Token Embedding + Position Embedding
  → TransformerDecoder (1 layer, 2 heads, d_model=32)
  → Linear head → vocab logits
  → CrossEntropyLoss with pad_token=0
```

- **Model type**: GPT-style decoder-only transformer
- **Output**: SequenceOutput (not ClassificationOutput) — proper for LM
- **Metric**: Perplexity (not Accuracy) — standard LM evaluation
- **Backend**: Flex on CPU (JIT-compiled), switch to Cuda for GPU

## Project Structure

```
src/
  main.rs        — entry point, backend selection, env_logger init
  tokenizer.rs   — BPE tokenizer with serde (vocab + merges persisted)
  data.rs        — LMExample, LMData (from_text), LMBatcher, LMBatch
  model.rs       — LMTransformer, LMTransformerConfig, forward()
  training.rs    — TrainStep, InferenceStep, run(), checkpoint/resume
data/
  train.txt      — raw training text (~40MB, 167K lines of Russian dialogs)
  tokenizer.json — cached BPE tokenizer (auto-generated on first run)
artifacts/
  model.bin      — trained model weights (CompactRecorder format)
  checkpoint.json — {"epochs_done": N, "total_epochs": M}
```

## Build & Run

```bash
# CPU (Flex backend — current)
cargo run --release

# GPU (RTX 3060 at home)
# Change Cargo.toml: features = ["cuda", "train"]
# Change main.rs: use burn::backend::{Autodiff, Cuda}; type MyBackend = Autodiff<Cuda>;
cargo run --release
```

## Training Pipeline (5 phases)

1. **Tokenizer** — trains BPE on first 500 lines (2000 merges), cached to `data/tokenizer.json`
2. **Dataset** — encodes first 10000 lines, chunks into 32-token sequences, 90/10 train/test split
3. **Model init** — `d_model=32, n_heads=2, n_layers=1, vocab=2000, max_len=32`
4. **Training** — `batch_size=16, num_workers=1, epochs=5, lr=5e-4 Noam scheduler`
5. **Save** — model weights + checkpoint.json with epoch progress

## Checkpoint / Resume

- `artifacts/checkpoint.json` tracks: `epochs_done`, `total_epochs` (auto-saved after `run()`)
- On restart: reads `checkpoint.json`, calculates `remaining = total - done`, trains only remaining
- Model weights loaded from `artifacts/model` (saved at end of each `run()`)
- Adam optimizer starts fresh each run (momentum rebuilds quickly)
- Delete `artifacts/checkpoint.json` to force full retraining from epoch 1

## Key Burn 0.21 API Notes

### Tensor creation from flat data
```rust
// CORRECT: create 1D tensor first, then reshape
Tensor::<B, 1, Int>::from_data(vec.as_slice(), device).reshape([batch, seq_len])

// WRONG: 2D from flat data causes rank mismatch panic
Tensor::<B, 2, Int>::from_data(vec.as_slice(), device) // PANICS
```

### TrainStep uses associated types (not generics)
```rust
impl<B: AutodiffBackend> TrainStep for MyModel<B> {
    type Input = MyBatch<B>;    // NOT TrainStep<MyBatch<B>>
    type Output = SequenceOutput<B>;
    fn step(&self, item: Self::Input) -> TrainOutput<Self::Output> { ... }
}
```

### TransformerDecoder API
```rust
let mask: Tensor<B, 3, Bool> = Tensor::<B, 2, Int>::zeros([t, t], device)
    .triu(1).equal_elem(1)  // triu on Int → Bool
    .unsqueeze_dim(0).repeat_dim(0, b);
let input = TransformerDecoderInput::new(x, memory).target_mask_attn(mask);
let output: Tensor<B, 3> = self.blocks.forward(input); // returns Tensor directly
```

### CrossEntropyLoss — pad_tokens not ignore_index
```rust
CrossEntropyLossConfig::new()
    .with_pad_tokens(Some(vec![0]))  // NOT .ignore_index(-100)
    .init(&device)
```

### SequenceOutput for language modeling
```rust
SequenceOutput::new(loss, logits, None, targets)
// logits: [batch, seq_len, vocab] — 3D, no flattening
// targets: [batch, seq_len] — 2D
// predictions: None — auto argmax
// Supports: Perplexity, Accuracy, Loss, CER, WER metrics
```

### Tensor ownership — get dims BEFORE reshape
```rust
let [b, t, v] = logits.dims();  // capture BEFORE reshape moves the tensor
let flat = logits.reshape([b * t, v]);
```

### Recorder save/load uses PathBuf
```rust
CompactRecorder::new().record(record, PathBuf::from("path/model"));
CompactRecorder::new().load(PathBuf::from("path/model"), &device);
```

## Known Pitfalls

1. **Tokenizer vocab serde**: `#[serde(skip)]` on `vocab` field causes `vocab_size()=0` after deserialization. Vocab MUST be serialized (it's `HashMap<String, usize>` — serializes fine).

2. **BPE encode O(n²)**: `encode()` uses `Vec::splice` which is O(n) per merge. On 40MB text this takes forever. Always use subset for tokenizer training AND dataset building.

3. **Tokenizer merges serde**: `HashMap<(usize,usize),usize>` can't be JSON key. Convert to `Vec<((usize,usize),usize)>` via custom serialize/deserialize.

4. **Empty encode panic**: `ids.len() - 1` underflows when text contains chars not in vocab. Guard with `if ids.len() <= 1` and map unknown chars to `unk_token`.

5. **env_logger required**: Burn 0.21 uses `log::info!()` for training progress. Without `env_logger::init()`, no output during training. TUI renderer works independently (requires `tui` feature, enabled by default).

6. **Perplexity vs Accuracy**: For language models, use `PerplexityMetric` (lower=better), not `AccuracyMetric` (meaningless for next-token prediction).

## Dependencies

```toml
burn = { version = "0.21.0", features = ["flex", "train"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
env_logger = "0.11"
log = "0.4"
```

## Model Scaling Reference

| Config | d_model | layers | heads | Params | VRAM (bs16) |
|---|---|---|---|---|---|
| Current (tiny) | 32 | 1 | 2 | ~50K | <1GB |
| Small | 128 | 4 | 4 | ~2M | ~2GB |
| Medium | 256 | 6 | 8 | ~15M | ~6GB |
| Large (RTX 3060 max) | 512 | 8 | 8 | ~50M | ~11GB |

Modify `LMTransformerConfig` in `training.rs` `load_or_init()` to scale.
