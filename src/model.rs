use burn::{
    config::Config,
    module::Module,
    nn::{
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig,
    },
    tensor::{backend::Backend, Bool, Int, Tensor},
};

#[derive(Module, Debug)]
pub struct LMTransformer<B: Backend> {
    pub wte: Embedding<B>,
    pub wpe: Embedding<B>,
    pub blocks: TransformerEncoder<B>,
    pub norm: RmsNorm<B>,
    pub lm_head: Linear<B>,
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

        let blocks = TransformerEncoderConfig::new(
            self.d_model,
            4 * self.d_model,
            self.n_heads,
            self.n_layers,
        )
        .init(device);

        let norm = RmsNormConfig::new(self.d_model).init(device);

        let lm_head = LinearConfig::new(self.d_model, self.vocab_size).init(device);

        LMTransformer {
            wte,
            wpe,
            blocks,
            norm,
            lm_head,
            d_model: self.d_model,
        }
    }
}

impl<B: Backend> LMTransformer<B> {
    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = idx.dims();

        // Получаем device ДО того, как idx будет перемещён
        let device = idx.device();

        // Token embeddings
        let tok_emb = self.wte.forward(idx);

        // Position ids
        let pos = Tensor::arange(0..seq_len as i64, &device)
            .unsqueeze_dim(0)
            .repeat_dim(0, batch_size);

        let pos_emb = self.wpe.forward(pos);

        let x = tok_emb + pos_emb;

        // Causal mask
        let mask: Tensor<B, 3, Bool> = Tensor::<B, 2, Int>::zeros([seq_len, seq_len], &device)
            .triu(1)
            .equal_elem(1)
            .unsqueeze_dim(0)
            .repeat_dim(0, batch_size);

        let input = TransformerEncoderInput::new(x).mask_attn(mask);

        let x = self.blocks.forward(input);

        // Final norm + head
        let x = self.norm.forward(x);
        self.lm_head.forward(x)
    }
}
