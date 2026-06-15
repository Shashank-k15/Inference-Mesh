use std::collections::HashMap;

use libp2p::PeerId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServerState {
    Offline = 0,
    Joining = 1,
    Online = 2,
}

impl ServerState {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(ServerState::Offline),
            1 => Some(ServerState::Joining),
            2 => Some(ServerState::Online),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub state: ServerState,
    pub throughput: f64,
    pub start_block: Option<u32>,
    pub end_block: Option<u32>,
    pub public_name: Option<String>,
    pub version: Option<String>,
    pub network_rps: Option<f64>,
    pub forward_rps: Option<f64>,
    pub inference_rps: Option<f64>,
    pub adapters: Vec<String>,
    pub torch_dtype: Option<String>,
    pub quant_type: Option<String>,
    pub using_relay: Option<bool>,
    pub cache_tokens_left: Option<u32>,
    pub next_pings: Option<HashMap<String, f64>>,
}

impl ServerInfo {
    pub fn new(state: ServerState, throughput: f64) -> Self {
        ServerInfo {
            state,
            throughput,
            start_block: None,
            end_block: None,
            public_name: None,
            version: None,
            network_rps: None,
            forward_rps: None,
            inference_rps: None,
            adapters: Vec::new(),
            torch_dtype: None,
            quant_type: None,
            using_relay: None,
            cache_tokens_left: None,
            next_pings: None,
        }
    }

    pub fn with_blocks(mut self, start: u32, end: u32) -> Self {
        self.start_block = Some(start);
        self.end_block = Some(end);
        self
    }

    pub fn with_throughput_info(mut self, network: f64, forward: f64, inference: f64) -> Self {
        self.network_rps = Some(network);
        self.forward_rps = Some(forward);
        self.inference_rps = Some(inference);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub num_blocks: u32,
    pub repository: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RemoteModuleInfo {
    pub uid: String,
    pub servers: HashMap<PeerId, ServerInfo>,
}

impl RemoteModuleInfo {
    pub fn new(uid: String) -> Self {
        RemoteModuleInfo {
            uid,
            servers: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RemoteSpanInfo {
    pub peer_id: PeerId,
    pub start: usize,
    pub end: usize,
    pub server_info: ServerInfo,
}

impl RemoteSpanInfo {
    pub fn length(&self) -> usize {
        self.end - self.start
    }
}
