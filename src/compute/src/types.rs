use inferencemesh_protocol::Dtype;

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
