use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};

use crate::block::TransformerBlock;
use inferencemesh_protocol::Dtype as DimDtype;

pub struct ComputeEngine {
    device: Device,
    blocks: HashMap<u32, TransformerBlock>,
}

impl ComputeEngine {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            blocks: HashMap::new(),
        }
    }

    pub fn load_blocks(
        &mut self,
        model_path: &Path,
        start_layer: u32,
        end_layer: u32,
    ) -> Result<()> {
        for layer in start_layer..=end_layer {
            let block =
                TransformerBlock::load_from_safetensors(model_path, layer, &self.device)?;
            self.blocks.insert(layer, block);
        }
        Ok(())
    }

    pub fn load_test_block(&mut self, layer: u32) -> Result<()> {
        let block = TransformerBlock::create_test_block(&self.device)?;
        self.blocks.insert(layer, block);
        Ok(())
    }

    pub fn forward(
        &self,
        layer: u32,
        hidden_states: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let block = self
            .blocks
            .get(&layer)
            .ok_or_else(|| candle_core::Error::Msg(format!("layer {} not loaded", layer)))?;
        block.forward(hidden_states, mask)
    }

    pub fn process_terminal_pass(
        &self,
        start_layer: u32,
        end_layer: u32,
        data: &[u8],
        dtype: DimDtype,
        shape: &[u64],
        mask_data: Option<&[u8]>,
        mask_dtype: Option<DimDtype>,
        mask_shape: Option<&[u64]>,
    ) -> anyhow::Result<Vec<f32>> {
        let results = self.batched_process_terminal_pass(
            start_layer,
            end_layer,
            &[(data.to_vec(), dtype, shape.to_vec())],
            mask_data,
            mask_dtype,
            mask_shape,
        )?;
        Ok(results.into_iter().next().unwrap_or_default())
    }

    pub fn batched_process_terminal_pass(
        &self,
        start_layer: u32,
        end_layer: u32,
        entries: &[(Vec<u8>, DimDtype, Vec<u64>)],
        mask_data: Option<&[u8]>,
        mask_dtype: Option<DimDtype>,
        mask_shape: Option<&[u64]>,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        if entries.is_empty() {
            return Ok(vec![]);
        }

        // All entries must have the same dtype and shape for stacking
        let candle_dtype = match entries[0].1 {
            DimDtype::F32 => DType::F32,
            DimDtype::F16 => DType::F16,
            DimDtype::BF16 => DType::BF16,
        };

        let mut tensors: Vec<Tensor> = Vec::with_capacity(entries.len());
        for (data, _dtype, shape) in entries {
            let ushape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
            let t = Tensor::from_raw_buffer(data, candle_dtype, &ushape, &self.device)?;
            tensors.push(t);
        }
        let batch = Tensor::stack(&tensors.iter().collect::<Vec<_>>(), 0)?;

        let mask = match (mask_data, mask_dtype, mask_shape) {
            (Some(md), Some(mdt), Some(ms)) => {
                let mdt_candle = match mdt {
                    DimDtype::F32 => DType::F32,
                    DimDtype::F16 => DType::F16,
                    DimDtype::BF16 => DType::BF16,
                };
                let mshape: Vec<usize> = ms.iter().map(|&x| x as usize).collect();
                Some(Tensor::from_raw_buffer(md, mdt_candle, &mshape, &self.device)?)
            }
            _ => None,
        };

        let mut current = batch;
        for layer in start_layer..=end_layer {
            current = self.forward(layer, &current, mask.as_ref())?;
        }

        // Split back: [B, ...] -> B × [...]
        let b = current.dims()[0];
        let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(b);
        for i in 0..b {
            let slice = current.get(i)?;
            let flat = slice.flatten_all()?.to_device(&Device::Cpu)?;
            outputs.push(flat.to_vec1::<f32>()?);
        }

        Ok(outputs)
    }
}
