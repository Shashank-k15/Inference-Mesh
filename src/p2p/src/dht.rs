use anyhow::Result;
use libp2p::kad::{Quorum, QueryId, Record, RecordKey};
use libp2p::Multiaddr;
use tracing::{debug, info};

use crate::behaviour::InferenceBehaviour;
use crate::types::ServerInfo;

pub fn layer_key(layer: u32) -> RecordKey {
    RecordKey::new(&format!("layer_{:03}", layer))
}

fn layer_addr_key(layer: u32, peer_id: &libp2p::PeerId) -> RecordKey {
    let peer_bytes = peer_id.to_bytes();
    let peer_hex: String = peer_bytes.iter().map(|b| format!("{:02x}", b)).collect();
    RecordKey::new(&format!("addr_{}_{:03}", peer_hex, layer))
}

pub fn server_info_key(peer_id: &libp2p::PeerId) -> RecordKey {
    RecordKey::new(&format!("server_info_{}", peer_id))
}

pub fn put_worker_address(
    behaviour: &mut InferenceBehaviour,
    layer: u32,
    peer_id: &libp2p::PeerId,
    addr: &Multiaddr,
) -> Result<QueryId> {
    let key = layer_addr_key(layer, peer_id);
    let value = addr.to_string().into_bytes();
    let record = Record::new(key, value);
    info!("Storing worker address for layer {}: {}", layer, addr);
    Ok(behaviour.kademlia.put_record(record, Quorum::One)?)
}

pub fn get_worker_address(
    behaviour: &mut InferenceBehaviour,
    layer: u32,
    peer_id: &libp2p::PeerId,
) -> QueryId {
    let key = layer_addr_key(layer, peer_id);
    info!("Looking up worker address for layer: {}", layer);
    behaviour.kademlia.get_record(key)
}

#[allow(dead_code)]
pub fn start_bootstrap(behaviour: &mut InferenceBehaviour) -> Result<QueryId> {
    info!("Starting Kademlia bootstrap");
    Ok(behaviour.kademlia.bootstrap()?)
}

pub fn announce_layers(
    behaviour: &mut InferenceBehaviour,
    start_layer: u32,
    end_layer: u32,
    info: &ServerInfo,
    peer_id: &libp2p::PeerId,
) -> Result<()> {
    for layer in start_layer..=end_layer {
        let key = layer_key(layer);
        info!("Announcing layer key: {:?}", key);
        behaviour.kademlia.start_providing(key)?;
    }

    let skey = server_info_key(peer_id);
    let value = serde_json::to_vec(info)?;
    let record = Record::new(skey, value);
    debug!("Storing ServerInfo for peer {:?}", peer_id);
    behaviour.kademlia.put_record(record, Quorum::One)?;

    Ok(())
}

pub fn get_server_info(
    behaviour: &mut InferenceBehaviour,
    peer_id: &libp2p::PeerId,
) -> QueryId {
    let key = server_info_key(peer_id);
    behaviour.kademlia.get_record(key)
}

pub fn build_server_info_record(peer_id: &libp2p::PeerId, info: &ServerInfo) -> Record {
    let key = server_info_key(peer_id);
    let value = serde_json::to_vec(info).unwrap_or_default();
    Record::new(key, value)
}

pub fn parse_server_info(record: &Record) -> Option<ServerInfo> {
    serde_json::from_slice(&record.value).ok()
}

pub fn announce_layers_simple(
    behaviour: &mut InferenceBehaviour,
    start_layer: u32,
    end_layer: u32,
) -> Result<()> {
    for layer in start_layer..=end_layer {
        let key = layer_key(layer);
        info!("Announcing layer key: {:?}", key);
        behaviour.kademlia.start_providing(key)?;
    }
    Ok(())
}

pub fn find_providers_for_layer(
    behaviour: &mut InferenceBehaviour,
    layer: u32,
) -> QueryId {
    let key = layer_key(layer);
    info!("Looking up providers for layer key: {:?}", key);
    behaviour.kademlia.get_providers(key)
}

#[allow(dead_code)]
pub fn find_providers_for_layers(
    behaviour: &mut InferenceBehaviour,
    layers: &[u32],
) -> Vec<QueryId> {
    layers
        .iter()
        .map(|&layer| find_providers_for_layer(behaviour, layer))
        .collect()
}

