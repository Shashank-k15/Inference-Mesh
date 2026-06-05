use std::io;

use crate::inference_capnp;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F16,
    BF16,
    F32,
}

impl Dtype {
    pub fn to_capnp(self) -> inference_capnp::Dtype {
        match self {
            Dtype::F16 => inference_capnp::Dtype::F16,
            Dtype::BF16 => inference_capnp::Dtype::Bf16,
            Dtype::F32 => inference_capnp::Dtype::F32,
        }
    }

    pub fn from_capnp(cd: inference_capnp::Dtype) -> Self {
        match cd {
            inference_capnp::Dtype::F16 => Dtype::F16,
            inference_capnp::Dtype::Bf16 => Dtype::BF16,
            inference_capnp::Dtype::F32 => Dtype::F32,
        }
    }

    pub fn from_capnp_res(
        cd: Result<inference_capnp::Dtype, capnp::NotInSchema>,
    ) -> io::Result<Self> {
        match cd {
            Ok(d) => Ok(Self::from_capnp(d)),
            Err(_) => Err(io::Error::new(io::ErrorKind::InvalidData, "unknown dtype")),
        }
    }

    pub fn element_size(&self) -> usize {
        match self {
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::F32 => 4,
        }
    }
}
