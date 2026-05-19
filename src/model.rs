use burn::{
    config::Config,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        transformer::{TransformerDecoder, TransformerDecoderConfig, TransformerDecoderInput},
    },
    tensor::{backend::Backend, Bool, Int, Tensor},
};

/// Transformer‑decoder language model (decoder‑only, like GPT).
#[derive(Module, Debug)]
pub struct LMTransformer<B: Backend> {
    pub wte: Embedding<B>,       // token embeddings
    pub wpe: Embedding<B>,       // position embeddings
    pub blocks: TransformerDecoder<B>,
    pub lm_head: Linear<B>,      // projects to vocab logits
    pub d_model: usize,
}

#[derive(Config, Debug)]
pub struct LMTransformerConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub max_len: usize,
}

impl LMTransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LMTransformer<B> {
        let wte = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let wpe = EmbeddingConfig::new(self.max_len, self.d_model).init(device);
        let lm_head = LinearConfig::new(self.d_model, self.vocab_size).init(device);

        let blocks = TransformerDecoderConfig::new(
            self.d_model,
            4 * self.d_model,
            self.n_heads,
            self.n_layers,
        )
        .init(device);

        LMTransformer {
            wte,
            wpe,
            blocks,
            lm_head,
            d_model: self.d_model,
        }
    }
}

impl<B: Backend> LMTransformer<B> {
    /// Forward pass: returns logits of shape [batch, seq_len, vocab_size].
    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, t] = idx.dims();
        let device = &self.devices()[0];
        let idx = idx.to_device(device);

        // position ids: 0..t-1 for each batch element
        let pos = Tensor::arange(0..t as i64, device)
            .reshape([1, t])
            .repeat_dim(0, b);

        let tok_emb = self.wte.forward(idx);
        let pos_emb = self.wpe.forward(pos);
        let x = (tok_emb + pos_emb) / 2.0;

        // Causal mask: upper triangle (above diagonal) = true = masked
        // triu works on Int tensors; convert to Bool via equal_elem
        let mask: Tensor<B, 3, Bool> = Tensor::<B, 2, Int>::zeros([t, t], device)
            .triu(1)
            .equal_elem(1)
            .unsqueeze_dim(0)
            .repeat_dim(0, b);

        // decoder-only: x serves as both target and memory
        let memory = x.clone();
        let input = TransformerDecoderInput::new(x, memory)
            .target_mask_attn(mask);

        // TransformerDecoder::forward returns Tensor<B, 3> directly
        let x = self.blocks.forward(input);

        // project to vocabulary
        self.lm_head.forward(x)
    }
}
