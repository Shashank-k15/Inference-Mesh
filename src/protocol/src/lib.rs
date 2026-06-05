pub mod codec;
pub mod request;
pub mod response;
pub mod types;

pub use codec::{ForwardPassCodec, ForwardPassProtocol};
pub use request::ForwardPassRequest;
pub use response::ForwardPassResponse;
pub use types::Dtype;

pub mod inference_capnp {
    include!(concat!(env!("OUT_DIR"), "/schema/inference_capnp.rs"));
}
