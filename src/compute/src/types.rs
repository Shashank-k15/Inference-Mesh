use anyhow::{Context, Result};
use inferencemesh_protocol::Dtype;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct TensorView {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

impl TensorView {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }

    pub fn byte_len(&self) -> usize {
        self.num_elements() * self.dtype.element_size()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(alias = "num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(alias = "num_key_value_heads", default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(alias = "num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(alias = "rms_norm_eps", default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(alias = "rope_theta", default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(alias = "max_position_embeddings", default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default = "default_max_position_embeddings_val")]
    pub max_position_embeddings_fallback: usize,
    #[serde(alias = "model_type", default)]
    pub model_type: Option<String>,
}

fn default_rms_norm_eps() -> f64 { 1e-5 }
fn default_rope_theta() -> f64 { 10000.0 }
fn default_max_position_embeddings_val() -> usize { 8192 }

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn num_key_value_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads()
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings.unwrap_or(self.max_position_embeddings_fallback)
    }

    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config from {}", path.display()))?;
        let config: ModelConfig = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse config from {}", path.display()))?;
        Ok(config)
    }
}

#[derive(Debug, Clone)]
pub struct RoPE {
    head_dim: usize,
    theta: f64,
}

impl RoPE {
    pub fn new(head_dim: usize, theta: f64, _max_position: usize) -> Self {
        RoPE { head_dim, theta }
    }

    pub fn from_config(config: &ModelConfig) -> Self {
        RoPE::new(config.head_dim(), config.rope_theta, config.max_position_embeddings())
    }

    pub fn apply(
        &self,
        q: &candle_core::Tensor,
        k: &candle_core::Tensor,
        position_ids: &candle_core::Tensor,
        device: &candle_core::Device,
    ) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
        use candle_core::Tensor;
        let seq_len = q.dim(2)?;
        let head_dim = self.head_dim;

        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0f32 / (self.theta.powf(i as f64 / head_dim as f64) as f32))
            .collect();
        let inv_freq_t = Tensor::new(inv_freq.as_slice(), device)?.unsqueeze(0)?;

        let pos_ids = position_ids.to_dtype(candle_core::DType::F32)?;
        let pos_flat = pos_ids.reshape(((), seq_len, 1))?;

        let freqs = pos_flat.broadcast_matmul(&inv_freq_t)?;
        let emb = Tensor::cat(&[&freqs, &freqs], 2)?;
        let emb = emb.unsqueeze(1)?;

        let cos = emb.cos()?;
        let sin = emb.sin()?;

        let cos_q = cos.broadcast_as(q.shape())?;
        let sin_q = sin.broadcast_as(q.shape())?;
        let cos_k = cos.broadcast_as(k.shape())?;
        let sin_k = sin.broadcast_as(k.shape())?;

        fn rotate_half(x: &Tensor) -> candle_core::Result<Tensor> {
            let last_dim = x.dim(x.dims().len() - 1)?;
            let x1 = x.narrow(x.dims().len() - 1, 0, last_dim / 2)?;
            let x2 = x.narrow(x.dims().len() - 1, last_dim / 2, last_dim - last_dim / 2)?;
            let neg_x2 = x2.neg()?;
            Tensor::cat(&[&neg_x2, &x1], x.dims().len() - 1)
        }

        let q_embed = ((q * &cos_q)? + (rotate_half(q)? * &sin_q)?)?;
        let k_embed = ((k * &cos_k)? + (rotate_half(k)? * &sin_k)?)?;
        Ok((q_embed, k_embed))
    }
}
