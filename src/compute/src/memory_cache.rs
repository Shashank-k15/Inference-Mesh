use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use candle_core::{Device, Tensor};

pub type Handle = u64;

pub struct MemoryCache {
    max_size_bytes: usize,
    current_size_bytes: AtomicU64,
    tensors: Mutex<HashMap<Handle, Tensor>>,
    handle_counter: AtomicU64,
    device: Device,
}

impl MemoryCache {
    pub fn new(max_size_bytes: usize, device: Device) -> Self {
        MemoryCache {
            max_size_bytes,
            current_size_bytes: AtomicU64::new(0),
            tensors: Mutex::new(HashMap::new()),
            handle_counter: AtomicU64::new(1),
            device,
        }
    }

    pub fn allocate(&self, shape: &[usize], dtype: candle_core::DType) -> Result<Handle, String> {
        let size_bytes: usize = shape.iter().product::<usize>()
            * dtype.size_in_bytes();
        let current = self.current_size_bytes.load(Ordering::SeqCst) as usize;

        if current + size_bytes > self.max_size_bytes {
            return Err(format!(
                "Cache full: needed {} bytes, available={}",
                size_bytes,
                self.max_size_bytes.saturating_sub(current)
            ));
        }

        self.current_size_bytes
            .fetch_add(size_bytes as u64, Ordering::SeqCst);
        let handle = self.handle_counter.fetch_add(1, Ordering::SeqCst);

        let tensor = Tensor::zeros(shape, dtype, &self.device)
            .map_err(|e| format!("Failed to create cache tensor: {}", e))?;

        self.tensors.lock().unwrap().insert(handle, tensor);
        Ok(handle)
    }

    pub fn free(&self, handle: Handle) {
        let mut tensors = self.tensors.lock().unwrap();
        if let Some(_tensor) = tensors.remove(&handle) {
            let size = _tensor.elem_count() * _tensor.dtype().size_in_bytes();
            self.current_size_bytes
                .fetch_sub(size as u64, Ordering::SeqCst);
        }
    }

    pub fn get(&self, handle: Handle) -> Option<Tensor> {
        self.tensors.lock().unwrap().get(&handle).cloned()
    }

    pub fn bytes_left(&self) -> usize {
        self.max_size_bytes
            .saturating_sub(self.current_size_bytes.load(Ordering::SeqCst) as usize)
    }
}
