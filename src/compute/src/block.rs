use std::path::Path;

use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{Module, VarBuilder};

use crate::types::{ModelConfig, RoPE};

#[derive(Debug)]
pub struct TransformerBlock {
    pub input_layernorm: candle_nn::RmsNorm,
    pub post_attention_layernorm: candle_nn::RmsNorm,
    pub q_proj: candle_nn::Linear,
    pub k_proj: candle_nn::Linear,
    pub v_proj: candle_nn::Linear,
    pub o_proj: candle_nn::Linear,
    pub gate_proj: candle_nn::Linear,
    pub up_proj: candle_nn::Linear,
    pub down_proj: candle_nn::Linear,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub n_kv_groups: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub eps: f32,
    pub rope: RoPE,
}

pub struct ForwardCache {
    pub residual1: Tensor,
    pub residual2: Tensor,
    pub normed_input: Tensor,
    pub attn_output: Tensor,
    pub normed_post: Tensor,
    pub gate_silu: Tensor,
    pub up_output: Tensor,
}

impl ForwardCache {
    pub fn new() -> Self {
        ForwardCache {
            residual1: Tensor::new(&[0f32], &Device::Cpu).unwrap(),
            residual2: Tensor::new(&[0f32], &Device::Cpu).unwrap(),
            normed_input: Tensor::new(&[0f32], &Device::Cpu).unwrap(),
            attn_output: Tensor::new(&[0f32], &Device::Cpu).unwrap(),
            normed_post: Tensor::new(&[0f32], &Device::Cpu).unwrap(),
            gate_silu: Tensor::new(&[0f32], &Device::Cpu).unwrap(),
            up_output: Tensor::new(&[0f32], &Device::Cpu).unwrap(),
        }
    }
}

impl TransformerBlock {
    pub fn create_test_block(device: &Device) -> Result<Self> {
        use candle_nn::VarMap;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let config = ModelConfig {
            hidden_size: 128,
            intermediate_size: 256,
            num_attention_heads: 4,
            num_key_value_heads: Some(4),
            num_hidden_layers: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: Some(2048),
            max_position_embeddings_fallback: 2048,
            model_type: Some("llama".into()),
        };
        Self::load_block_from_vb(vb, &config, device)
    }

    pub fn load_from_safetensors(
        model_path: &Path,
        config: &ModelConfig,
        layer_idx: u32,
        device: &Device,
    ) -> Result<Self> {
        let mut path = model_path.to_path_buf();
        if path.is_dir() {
            for name in &["model.safetensors", "model.safetensors.index.json"] {
                let c = path.join(name);
                if c.exists() {
                    path = c;
                    break;
                }
            }
            if path.is_dir() {
                if let Ok(entries) = std::fs::read_dir(&path) {
                    for entry in entries.flatten() {
                        let n = entry.file_name();
                        let n = n.to_string_lossy();
                        if n.ends_with(".safetensors") && !n.contains("index.json") {
                            path = entry.path();
                            break;
                        }
                    }
                }
            }
        }

        let data = std::fs::read(&path)
            .map_err(|e| candle_core::Error::Msg(format!("read model: {e}")))?;
        let vb = VarBuilder::from_buffered_safetensors(data, DType::F32, device)?;
        Self::load_block_from_vb(vb.pp(format!("model.layers.{layer_idx}")), config, device)
    }

    pub fn load_block_from_vb(vb: VarBuilder, config: &ModelConfig, _device: &Device) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads();
        let n_kv_groups = config.num_key_value_groups();
        let head_dim = config.head_dim();
        let eps = config.rms_norm_eps as f32;
        let rope = RoPE::from_config(config);

        let input_layernorm = candle_nn::rms_norm(hidden_size, eps as f64, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            candle_nn::rms_norm(hidden_size, eps as f64, vb.pp("post_attention_layernorm"))?;
        let q_proj = Self::linear_no_bias(hidden_size, n_heads * head_dim, vb.pp("self_attn.q_proj"))?;
        let k_proj = Self::linear_no_bias(hidden_size, n_kv_heads * head_dim, vb.pp("self_attn.k_proj"))?;
        let v_proj = Self::linear_no_bias(hidden_size, n_kv_heads * head_dim, vb.pp("self_attn.v_proj"))?;
        let o_proj = Self::linear_no_bias(n_heads * head_dim, hidden_size, vb.pp("self_attn.o_proj"))?;
        let gate_proj = Self::linear_no_bias(hidden_size, intermediate_size, vb.pp("mlp.gate_proj"))?;
        let up_proj = Self::linear_no_bias(hidden_size, intermediate_size, vb.pp("mlp.up_proj"))?;
        let down_proj = Self::linear_no_bias(intermediate_size, hidden_size, vb.pp("mlp.down_proj"))?;

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            up_proj,
            down_proj,
            n_heads,
            n_kv_heads,
            n_kv_groups,
            head_dim,
            hidden_size,
            eps,
            rope,
        })
    }

    fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<candle_nn::Linear> {
        let ws = vb.get((out_dim, in_dim), "weight")?;
        let bs = candle_core::Tensor::zeros(out_dim, ws.dtype(), ws.device())?;
        Ok(candle_nn::Linear::new(ws, Some(bs)))
    }

    fn compute_attention_mask(
        &self,
        b_times_heads: usize,
        seq_len: usize,
        past_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let total_len = past_len + seq_len;
        let mut mask_data: Vec<f32> = Vec::with_capacity(b_times_heads * total_len * total_len);
        for _b in 0..b_times_heads {
            for i in 0..total_len {
                for j in 0..total_len {
                    mask_data.push(if j > i { f32::NEG_INFINITY } else { 0.0 });
                }
            }
        }
        let full_mask =
            Tensor::from_vec(mask_data, (b_times_heads, total_len, total_len), device)?;

        if past_len > 0 {
            full_mask.narrow(1, past_len, seq_len)
        } else {
            Ok(full_mask)
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden) = hidden_states.dims3()?;
        let device = hidden_states.device();

        let position_ids = Tensor::arange(0u32, seq_len as u32, device)?
            .unsqueeze(0)?
            .repeat((b_sz, 1))?;

        self.forward_inner(hidden_states, mask, &position_ids, None)
            .map(|(output, _kv)| output)
    }

    pub fn forward_inner(
        &self,
        hidden_states: &Tensor,
        mask: Option<&Tensor>,
        position_ids: &Tensor,
        layer_past: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let (b_sz, seq_len, _hidden) = hidden_states.dims3()?;
        let device = hidden_states.device();
        let residual = hidden_states.clone();

        let normed =
            candle_nn::ops::rms_norm(hidden_states, self.input_layernorm.weight(), self.eps)?;
        let q = self.q_proj.forward(&normed)?;
        let k = self.k_proj.forward(&normed)?;
        let v = self.v_proj.forward(&normed)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = {
            self.rope.apply(&q, &k, position_ids, device)?
        };

        let past_len = if let Some((past_k, past_v)) = layer_past {
            let pk = past_k;
            let pv = past_v;
            let k_full = Tensor::cat(&[pk, &k], 2)?;
            let v_full = Tensor::cat(&[pv, &v], 2)?;
            let pl = pk.dim(2)?;
            (k_full, v_full, pl)
        } else {
            (k, v, 0)
        };

        let (k_full, v_full, past_len_val) = past_len;

        let k_flat = k_full.reshape((b_sz * self.n_kv_heads, past_len_val + seq_len, self.head_dim))?;
        let v_flat = v_full.reshape((b_sz * self.n_kv_heads, past_len_val + seq_len, self.head_dim))?;

        let q_flat = q.reshape((b_sz * self.n_heads, seq_len, self.head_dim))?;
        let k_repeated = repeat_kv(&k_flat, self.n_kv_groups)?;
        let v_repeated = repeat_kv(&v_flat, self.n_kv_groups)?;

        let att =
            (q_flat.matmul(&k_repeated.t()?.contiguous()?)? / (self.head_dim as f64).sqrt())?;

        let causal_mask = self.compute_attention_mask(b_sz * self.n_heads, seq_len, past_len_val, device)?;
        let att = if let Some(user_mask) = mask {
            let m = user_mask.broadcast_as(att.shape())?;
            (att + &causal_mask + m)?
        } else {
            (att + &causal_mask)?
        };

        let att = candle_nn::ops::softmax_last_dim(&att)?;

        let attn_output = att.matmul(&v_repeated)?;
        let attn_output =
            attn_output.reshape((b_sz, self.n_heads, seq_len, self.head_dim))?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, seq_len, self.hidden_size))?;
        let attn_output = self.o_proj.forward(&attn_output)?;
        let hidden_states = (residual + attn_output)?;

        let residual = hidden_states.clone();
        let normed = candle_nn::ops::rms_norm(
            &hidden_states,
            self.post_attention_layernorm.weight(),
            self.eps,
        )?;

        let gate = self.gate_proj.forward(&normed)?.silu()?;
        let up = self.up_proj.forward(&normed)?;
        let mlp_output = self.down_proj.forward(&(gate * up)?)?;

        let hidden_states = (residual + mlp_output)?;

        let new_kv = Some((
            k_full.reshape((b_sz, self.n_kv_heads, past_len_val + seq_len, self.head_dim))?,
            v_full.reshape((b_sz, self.n_kv_heads, past_len_val + seq_len, self.head_dim))?,
        ));

        Ok((hidden_states, new_kv))
    }

    pub fn forward_with_kv_cache(
        &self,
        hidden_states: &Tensor,
        mask: Option<&Tensor>,
        position_ids: &Tensor,
        layer_past: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        self.forward_inner(hidden_states, mask, position_ids, layer_past)
    }

    pub fn backward(
        &self,
        _hidden_states: &Tensor,
        _mask: Option<&Tensor>,
        _grad_output: &Tensor,
    ) -> Result<Tensor> {
        Err(candle_core::Error::Msg(
            "Manual backward pass not yet implemented. Use VarMap-based autograd for gradient computation.".into(),
        ))
    }
}

pub fn repeat_kv(kv: &Tensor, n_groups: usize) -> Result<Tensor> {
    if n_groups == 1 {
        return Ok(kv.clone());
    }
    let dims = kv.shape().dims().to_vec();
    let b_times_kv_heads = dims[0];
    let seq_len = dims[1];
    let head_dim = dims[2];

    let kv = kv.unsqueeze(1)?;
    let kv = kv.expand((b_times_kv_heads, n_groups, seq_len, head_dim))?;
    kv.reshape((b_times_kv_heads * n_groups, seq_len, head_dim))
}
