use std::path::Path;

use candle_core::{Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};

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
    pub head_dim: usize,
    pub hidden_size: usize,
    pub eps: f32,
}

impl TransformerBlock {
    pub fn create_test_block(device: &Device) -> Result<Self> {
        use candle_nn::VarMap;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);
        Self::load_block_from_vb(vb, device)
    }

    pub fn load_from_safetensors(
        model_path: &Path,
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
        let vb = VarBuilder::from_buffered_safetensors(data, candle_core::DType::F32, device)?;
        Self::load_block_from_vb(vb.pp(format!("model.layers.{layer_idx}")), device)
    }

    pub fn load_block_from_vb(vb: VarBuilder, _device: &Device) -> Result<Self> {
        let hidden_size = 128usize;
        let intermediate_size = 256usize;
        let n_heads = 4usize;
        let n_kv_heads = 4usize;
        let head_dim = 32usize;
        let eps = 1e-5f32;

        let input_layernorm = candle_nn::rms_norm(hidden_size, eps as f64, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            candle_nn::rms_norm(hidden_size, eps as f64, vb.pp("post_attention_layernorm"))?;
        let q_proj = candle_nn::linear(hidden_size, n_heads * head_dim, vb.pp("self_attn.q_proj"))?;
        let k_proj = candle_nn::linear(hidden_size, n_kv_heads * head_dim, vb.pp("self_attn.k_proj"))?;
        let v_proj = candle_nn::linear(hidden_size, n_kv_heads * head_dim, vb.pp("self_attn.v_proj"))?;
        let o_proj = candle_nn::linear(n_heads * head_dim, hidden_size, vb.pp("self_attn.o_proj"))?;
        let gate_proj = candle_nn::linear(hidden_size, intermediate_size, vb.pp("mlp.gate_proj"))?;
        let up_proj = candle_nn::linear(hidden_size, intermediate_size, vb.pp("mlp.up_proj"))?;
        let down_proj = candle_nn::linear(intermediate_size, hidden_size, vb.pp("mlp.down_proj"))?;

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
            head_dim,
            hidden_size,
            eps,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden) = hidden_states.dims3()?;
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

        let q = q.reshape((b_sz * self.n_heads, seq_len, self.head_dim))?;
        let k = k.reshape((b_sz * self.n_kv_heads, seq_len, self.head_dim))?;
        let v = v.reshape((b_sz * self.n_kv_heads, seq_len, self.head_dim))?;

        let att =
            (q.matmul(&k.t()?.contiguous()?)? / (self.head_dim as f64).sqrt())?;

        let att = if let Some(mask) = mask {
            (att + mask)?
        } else {
            att
        };

        let att = candle_nn::ops::softmax_last_dim(&att)?;

        let attn_output = att.matmul(&v)?;
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
        Ok(hidden_states)
    }
}
