use anyhow::Result;
use libp2p::kad::{Quorum, QueryId, Record, RecordKey};
use libp2p::Multiaddr;
use tracing::info;

use crate::behaviour::InferenceBehaviour;

pub fn layer_key(layer: u32) -> RecordKey {
    RecordKey::new(&format!("layer_{:03}", layer))
}

fn layer_addr_key(layer: u32, peer_id: &libp2p::PeerId) -> RecordKey {
    RecordKey::new(&format!("addr_{}_{:03}", peer_id, layer))
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
