use std::fmt;
use std::io;

use capnp::message::ReaderOptions;

use crate::inference_capnp;
use crate::types::Dtype;

fn ce2io(e: capnp::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e)
}

pub struct ForwardPassResponse {
    bytes: Vec<u8>,
    msg: capnp::message::Reader<capnp::serialize::OwnedSegments>,
}

impl fmt::Debug for ForwardPassResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ForwardPassResponse")
            .field("len", &self.bytes.len())
            .finish()
    }
}

impl ForwardPassResponse {
    pub fn from_bytes(bytes: Vec<u8>) -> io::Result<Self> {
        let mut cursor = std::io::Cursor::new(&bytes[..]);
        let msg =
            capnp::serialize_packed::read_message(&mut cursor, ReaderOptions::default())
                .map_err(ce2io)?;
        Ok(Self { bytes, msg })
    }

    pub fn build(
        request_id: u64,
        dtype: Dtype,
        shape: &[u64],
        data: &[u8],
    ) -> io::Result<Self> {
        let mut message = capnp::message::Builder::new_default();
        let mut builder =
            message.init_root::<inference_capnp::forward_pass_response::Builder>();

        builder.set_request_id(request_id);

        {
            let mut tensor = builder.reborrow().init_tensor();
            tensor.set_dtype(dtype.to_capnp());
            let mut sl = tensor.reborrow().init_shape(shape.len() as u32);
            for (i, &s) in shape.iter().enumerate() {
                sl.set(i as u32, s);
            }
            let db = tensor.reborrow().init_data(data.len() as u32);
            (&mut *db).copy_from_slice(data);
        }

        let mut bytes = Vec::new();
        capnp::serialize_packed::write_message(&mut bytes, &message).map_err(ce2io)?;
        Self::from_bytes(bytes)
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    fn root(&self) -> io::Result<inference_capnp::forward_pass_response::Reader<'_>> {
        self.msg
            .get_root::<inference_capnp::forward_pass_response::Reader<'_>>()
            .map_err(ce2io)
    }

    pub fn request_id(&self) -> io::Result<u64> {
        Ok(self.root()?.get_request_id())
    }

    pub fn tensor_dtype(&self) -> io::Result<Dtype> {
        Dtype::from_capnp_res(self.root()?.get_tensor().map_err(ce2io)?.get_dtype())
    }

    pub fn tensor_shape(&self) -> io::Result<Vec<u64>> {
        let list = self
            .root()?
            .get_tensor()
            .map_err(ce2io)?
            .get_shape()
            .map_err(ce2io)?;
        let mut shape = Vec::with_capacity(list.len() as usize);
        for i in 0..list.len() {
            shape.push(list.get(i));
        }
        Ok(shape)
    }

    pub fn tensor_data(&self) -> io::Result<Vec<u8>> {
        let root = self.root()?;
        let tensor = root.get_tensor().map_err(ce2io)?;
        let data: &[u8] = tensor.get_data().map_err(ce2io)?;
        Ok(data.to_vec())
    }
}
