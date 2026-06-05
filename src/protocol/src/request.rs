use std::fmt;
use std::io;

use capnp::message::ReaderOptions;
use libp2p::PeerId;

use crate::inference_capnp;
use crate::types::Dtype;

fn ce2io(e: capnp::Error) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e)
}

pub struct ForwardPassRequest {
    bytes: Vec<u8>,
    msg: capnp::message::Reader<capnp::serialize::OwnedSegments>,
}

impl fmt::Debug for ForwardPassRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ForwardPassRequest")
            .field("len", &self.bytes.len())
            .finish()
    }
}

impl ForwardPassRequest {
    pub fn from_bytes(bytes: Vec<u8>) -> io::Result<Self> {
        let mut cursor = std::io::Cursor::new(&bytes[..]);
        let msg =
            capnp::serialize_packed::read_message(&mut cursor, ReaderOptions::default())
                .map_err(ce2io)?;
        Ok(Self { bytes, msg })
    }

    fn build_bytes(
        route: &[PeerId],
        hop_index: u32,
        request_id: u64,
        dtype: Dtype,
        shape: &[u64],
        data: &[u8],
        mask: Option<(Dtype, &[u64], &[u8])>,
    ) -> io::Result<Vec<u8>> {
        let mut message = capnp::message::Builder::new_default();
        let mut builder = message.init_root::<inference_capnp::forward_pass_request::Builder>();

        let mut route_list = builder.reborrow().init_route(route.len() as u32);
        for (i, peer) in route.iter().enumerate() {
            route_list.set(i as u32, peer.to_bytes().as_slice());
        }

        builder.set_hop_index(hop_index);
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

        if let Some((md, ms, md2)) = mask {
            let mut mb = builder.reborrow().init_mask();
            mb.set_dtype(md.to_capnp());
            let mut sl = mb.reborrow().init_shape(ms.len() as u32);
            for (i, &s) in ms.iter().enumerate() {
                sl.set(i as u32, s);
            }
            let db = mb.reborrow().init_data(md2.len() as u32);
            (&mut *db).copy_from_slice(md2);
        }

        let mut buf = Vec::new();
        capnp::serialize_packed::write_message(&mut buf, &message).map_err(ce2io)?;
        Ok(buf)
    }

    pub fn build(
        route: &[PeerId],
        hop_index: u32,
        request_id: u64,
        dtype: Dtype,
        shape: &[u64],
        data: &[u8],
        mask: Option<(Dtype, &[u64], &[u8])>,
    ) -> io::Result<Self> {
        let bytes = Self::build_bytes(route, hop_index, request_id, dtype, shape, data, mask)?;
        Self::from_bytes(bytes)
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    fn root(&self) -> io::Result<inference_capnp::forward_pass_request::Reader<'_>> {
        self.msg
            .get_root::<inference_capnp::forward_pass_request::Reader<'_>>()
            .map_err(ce2io)
    }

    pub fn route(&self) -> io::Result<Vec<PeerId>> {
        let list = self.root()?.get_route().map_err(ce2io)?;
        let mut peers = Vec::with_capacity(list.len() as usize);
        for i in 0..list.len() {
            let data_reader = list.get(i).map_err(ce2io)?;
            let bytes = data_reader.as_ref();
            let peer = PeerId::from_bytes(bytes).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "invalid PeerId in route")
            })?;
            peers.push(peer);
        }
        Ok(peers)
    }

    pub fn hop_index(&self) -> io::Result<u32> {
        Ok(self.root()?.get_hop_index())
    }

    pub fn request_id(&self) -> io::Result<u64> {
        Ok(self.root()?.get_request_id())
    }

    pub fn is_terminal(&self) -> io::Result<bool> {
        let route = self.route()?;
        Ok(self.hop_index()? as usize == route.len().saturating_sub(1))
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

    pub fn mask_data(&self) -> io::Result<Option<(Dtype, Vec<u64>, Vec<u8>)>> {
        let root = self.root()?;
        if !root.has_mask() {
            return Ok(None);
        }
        let mask = root.get_mask().map_err(ce2io)?;
        let dtype = Dtype::from_capnp_res(mask.get_dtype())?;
        let sl = mask.get_shape().map_err(ce2io)?;
        let mut shape = Vec::with_capacity(sl.len() as usize);
        for i in 0..sl.len() {
            shape.push(sl.get(i));
        }
        let data: &[u8] = mask.get_data().map_err(ce2io)?.as_ref();
        Ok(Some((dtype, shape, data.to_vec())))
    }
}
