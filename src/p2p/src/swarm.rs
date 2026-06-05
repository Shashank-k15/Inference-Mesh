use std::collections::{HashMap, HashSet};
use std::time::Duration;

use anyhow::{anyhow, Result};
use futures::StreamExt;
use libp2p::{
    kad::{self, QueryId},
    noise, tcp, yamux,
    request_response::{self, OutboundRequestId, ResponseChannel},
    swarm::SwarmEvent,
    Multiaddr, PeerId, Swarm, SwarmBuilder,
};
use tracing::{debug, error, info, warn};

use crate::behaviour::{InferenceBehaviour, InferenceBehaviourEvent};
use crate::dht;
use crate::provider_cache::ProviderCache;
#[cfg(feature = "compute")]
use inferencemesh_compute::ComputeEngine;
use inferencemesh_protocol::{
    Dtype, ForwardPassRequest, ForwardPassResponse,
};
use crate::routing;

pub fn build_swarm(
    keypair: &libp2p::identity::Keypair,
) -> Result<Swarm<InferenceBehaviour>> {
    let swarm = SwarmBuilder::with_existing_identity(keypair.clone())
        .with_tokio()
        .with_tcp(
            tcp::Config::default().nodelay(true),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_quic()
        .with_behaviour(|_key| InferenceBehaviour::new(keypair))
        .expect("behaviour setup should not fail")
        .with_swarm_config(|cfg| {
            cfg.with_idle_connection_timeout(Duration::from_secs(60))
        })
        .build();

    Ok(swarm)
}

pub async fn run_bootstrap(
    swarm: &mut Swarm<InferenceBehaviour>,
    port: u16,
) -> Result<()> {
    let listen_addr: Multiaddr = format!("/ip4/0.0.0.0/tcp/{}", port).parse()?;
    swarm.listen_on(listen_addr.clone())?;
    info!("Bootstrap node listening on {}", listen_addr);

    let quic_listen: Multiaddr = format!("/ip4/0.0.0.0/udp/{}/quic-v1", port).parse()?;
    swarm.listen_on(quic_listen.clone())?;
    info!("Bootstrap node listening on {}", quic_listen);

    loop {
        let event = swarm.select_next_some().await;
        match event {
            SwarmEvent::Behaviour(InferenceBehaviourEvent::Kademlia(e)) => {
                handle_kad_event(e);
            }
            SwarmEvent::NewListenAddr { address, .. } => {
                info!("Bootstrap: new listen addr: {}", address);
            }
            SwarmEvent::ConnectionEstablished {
                peer_id,
                endpoint,
                ..
            } => {
                info!("Bootstrap: connection established with {:?}", peer_id);
                let remote_addr = endpoint.get_remote_address().clone();
                swarm.behaviour_mut().kademlia.add_address(&peer_id, remote_addr);
            }
            SwarmEvent::ConnectionClosed { peer_id, .. } => {
                info!("Bootstrap: connection closed with {:?}", peer_id);
            }
            e => {
                debug!("Bootstrap event: {:?}", e);
            }
        }
    }
}

struct PendingForward {
    upstream_channel: ResponseChannel<ForwardPassResponse>,
}

#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub max_latency_ms: u64,
    pub enabled: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_latency_ms: 5,
            enabled: false,
        }
    }
}

fn add_bootstrap_addr(
    swarm: &mut Swarm<InferenceBehaviour>,
    bootstrap_triggered: &mut bool,
    peer_id: PeerId,
    addr: Multiaddr,
) {
    if !*bootstrap_triggered {
        *bootstrap_triggered = true;
        swarm.behaviour_mut().kademlia.add_address(&peer_id, addr);
        if let Err(e) = swarm.behaviour_mut().kademlia.bootstrap() {
            warn!("Bootstrap error: {:?}", e);
        } else {
            info!("Kademlia bootstrap triggered via {:?}", peer_id);
        }
    }
}

/// Send an echo response (mirrors input tensor back to caller).
fn send_echo_response(
    swarm: &mut Swarm<InferenceBehaviour>,
    channel: ResponseChannel<ForwardPassResponse>,
    rid: u64,
    request: &ForwardPassRequest,
) -> Result<()> {
    info!("Worker: terminal hop, echoing payload back (no compute engine)");
    let response = ForwardPassResponse::build(
        rid,
        request.tensor_dtype()?,
        &request.tensor_shape()?,
        &request.tensor_data()?,
    )?;
    if let Err(e) = swarm
        .behaviour_mut()
        .request_response
        .send_response(channel, response)
    {
        error!("Worker: failed to send terminal response: {:?}", e);
    }
    Ok(())
}

pub async fn run_worker(
    swarm: &mut Swarm<InferenceBehaviour>,
    bootstrap_addrs: Vec<Multiaddr>,
    start_layer: u32,
    end_layer: u32,
    #[cfg(feature = "compute")] engine: Option<std::sync::Arc<ComputeEngine>>,
    peer_addresses: HashMap<PeerId, Multiaddr>,
    _batch_config: Option<BatchConfig>,
) -> Result<()> {
    // Dial bootstrap nodes so the worker can join the network.
    for addr in &bootstrap_addrs {
        swarm.dial(addr.clone())?;
        info!("Worker dialing bootstrap: {}", addr);
    }

    // Pre-register known peer addresses in Kademlia for direct connectivity.
    for (peer_id, addr) in &peer_addresses {
        swarm.behaviour_mut().kademlia.add_address(peer_id, addr.clone());
        info!("Worker: pre-registered peer {:?} at {}", peer_id, addr);
    }

    let mut bootstrapped = false;
    let mut bootstrap_triggered = false;
    let mut layers_announced = false;
    let mut pending_forwards: HashMap<OutboundRequestId, PendingForward> = HashMap::new();

    loop {
        let event = swarm.select_next_some().await;
        match event {
            SwarmEvent::Behaviour(InferenceBehaviourEvent::Kademlia(e)) => {
                handle_kad_event(e.clone());

                if !bootstrapped
                    && matches!(&e, kad::Event::RoutingUpdated { is_new_peer: true, .. })
                {
                    bootstrapped = true;
                    info!("Worker bootstrapped");
                }

                if bootstrapped && !layers_announced {
                    info!("Worker announcing layers {}-{}", start_layer, end_layer);
                    dht::announce_layers(swarm.behaviour_mut(), start_layer, end_layer)?;
                    layers_announced = true;
                }
            }
            SwarmEvent::NewListenAddr { address, .. } => {
                info!("Worker: listening on {}", address);
            }
            SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
                let remote_addr = endpoint.get_remote_address().clone();
                info!("Worker: connected to {:?} at {}", peer_id, remote_addr);
                add_bootstrap_addr(swarm, &mut bootstrap_triggered, peer_id, remote_addr);
            }
            SwarmEvent::Behaviour(InferenceBehaviourEvent::RequestResponse(e)) => {
                match e {
                    request_response::Event::Message {
                        peer,
                        message:
                            request_response::Message::Request { request, channel, .. },
                    } => {
                        let rid = request.request_id().unwrap_or(0);
                        let hop = request.hop_index().unwrap_or(0);
                        let route_len = request.route().map(|r| r.len()).unwrap_or(0);
                        info!(
                            "Worker: received forward pass id={} from {:?}, hop={}/{}",
                            rid, peer, hop, route_len
                        );

                        let is_terminal = request.is_terminal().unwrap_or(false);

                        if is_terminal {
                            handle_terminal_hop(
                                swarm,
                                channel,
                                rid,
                                &request,
                                start_layer,
                                end_layer,
                                #[cfg(feature = "compute")]
                                &engine,
                            )?;
                        } else {
                            match routing::build_next_request(&request) {
                                Ok((next_peer, next_request)) => {
                                    info!("Worker: forwarding to next hop {:?}", next_peer);
                                    let outbound_id = swarm
                                        .behaviour_mut()
                                        .request_response
                                        .send_request(&next_peer, next_request);
                                    pending_forwards.insert(
                                        outbound_id,
                                        PendingForward { upstream_channel: channel },
                                    );
                                }
                                Err(e) => {
                                    error!("Worker: failed to build next request: {:?}", e);
                                }
                            }
                        }
                    }
                    request_response::Event::Message {
                        message:
                            request_response::Message::Response { request_id, response, .. },
                        ..
                    } => {
                        if let Some(pending) = pending_forwards.remove(&request_id) {
                            if let Err(e) = swarm.behaviour_mut()
                                .request_response.send_response(pending.upstream_channel, response)
                            {
                                error!("Worker: failed to forward upstream response: {:?}", e);
                            }
                        }
                    }
                    request_response::Event::OutboundFailure {
                        request_id,
                        error,
                        ..
                    } => {
                        if let Some(pending) = pending_forwards.remove(&request_id) {
                            warn!("Worker: forwarded request failed: {:?}, sending empty failure signal upstream", error);
                            let failure = ForwardPassResponse::build(0, Dtype::F32, &[], &[])
                                .unwrap_or_else(|_| ForwardPassResponse::build(0, Dtype::F32, &[0], &[]).unwrap());
                            let _ = swarm.behaviour_mut()
                                .request_response.send_response(pending.upstream_channel, failure);
                        }
                    }
                    _ => debug!("Worker: other rr event: {:?}", e),
                }
            }
            SwarmEvent::ConnectionClosed { peer_id, .. } => {
                warn!("Worker: connection closed with {:?}", peer_id);
            }
            SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                warn!("Worker: outgoing connection error to {:?}: {:?}", peer_id, error);
            }
            e => debug!("Worker event: {:?}", e),
        }
    }
}

/// Handle a terminal hop in the inference pipeline.
///
/// When the compute feature is enabled and an engine is provided, runs the
/// actual forward pass through the loaded transformer blocks. Otherwise,
/// echoes the input tensor back (useful for network-only testing).
fn handle_terminal_hop(
    swarm: &mut Swarm<InferenceBehaviour>,
    channel: ResponseChannel<ForwardPassResponse>,
    rid: u64,
    request: &ForwardPassRequest,
    start_layer: u32,
    end_layer: u32,
    #[cfg(feature = "compute")] engine: &Option<std::sync::Arc<ComputeEngine>>,
) -> Result<()> {
    #[cfg(feature = "compute")]
    if let Some(ref eng) = engine {
        info!("Worker: terminal hop, running compute engine layers {}-{}", start_layer, end_layer);
        let result = eng.process_terminal_pass(
            start_layer,
            end_layer,
            &request.tensor_data()?,
            request.tensor_dtype()?,
            &request.tensor_shape()?,
            None,
            None,
            None,
        );
        match result {
            Ok(output_vec) => {
                let output_bytes: Vec<u8> = output_vec
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();
                let output_shape = vec![output_bytes.len() as u64 / 4];
                let response = ForwardPassResponse::build(
                    rid,
                    request.tensor_dtype().unwrap_or(Dtype::F32),
                    &output_shape,
                    &output_bytes,
                );
                match response {
                    Ok(resp) => {
                        let _ = swarm
                            .behaviour_mut()
                            .request_response
                            .send_response(channel, resp);
                    }
                    Err(e) => {
                        error!("Worker: failed to build compute response: {:?}", e);
                        let _ = swarm
                            .behaviour_mut()
                            .request_response
                            .send_response(channel, ForwardPassResponse::build(rid, Dtype::F32, &[], &[]).unwrap());
                    }
                }
            }
            Err(e) => {
                error!("Worker: compute failed: {:?}", e);
                let _ = swarm
                    .behaviour_mut()
                    .request_response
                    .send_response(channel, ForwardPassResponse::build(rid, Dtype::F32, &[], &[]).unwrap());
            }
        }
        return Ok(());
    }

    // No compute engine available — echo payload back.
    send_echo_response(swarm, channel, rid, request)
}

pub async fn run_client(
    swarm: &mut Swarm<InferenceBehaviour>,
    bootstrap_addrs: Vec<Multiaddr>,
    layers: Vec<u32>,
    payload: Vec<u8>,
    worker_addresses: HashMap<PeerId, Multiaddr>,
) -> Result<ForwardPassResponse> {
    for addr in &bootstrap_addrs {
        swarm.dial(addr.clone())?;
        info!("Client dialing bootstrap: {}", addr);
    }

    for (peer_id, addr) in &worker_addresses {
        swarm.behaviour_mut().kademlia.add_address(peer_id, addr.clone());
        info!("Client: pre-registered worker {:?} at {}", peer_id, addr);
    }

    let mut bootstrapped = false;
    let mut bootstrap_triggered = false;
    let mut queries_sent = false;
    let mut providers: HashMap<u32, PeerId> = HashMap::new();
    let mut pending_queries: HashMap<QueryId, u32> = HashMap::new();
    let mut chain_built = false;
    let mut result: Option<ForwardPassResponse> = None;
    let mut provider_cache = ProviderCache::new(Duration::from_secs(60));
    let mut failed_peers: HashSet<PeerId> = HashSet::new();
    let mut retry_count: u32 = 0;
    // Track the chain so we can identify the failing peer on outbound failure.
    let mut current_chain: Vec<PeerId> = Vec::new();
    const MAX_RETRIES: u32 = 3;

    loop {
        let event = swarm.select_next_some().await;
        match event {
            SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
                let remote_addr = endpoint.get_remote_address().clone();
                info!("Client: connected to {:?} at {}", peer_id, remote_addr);
                add_bootstrap_addr(swarm, &mut bootstrap_triggered, peer_id, remote_addr);
            }
            SwarmEvent::Behaviour(InferenceBehaviourEvent::Kademlia(e)) => {
                match e {
                    kad::Event::RoutingUpdated { is_new_peer: true, .. } => {
                        if !bootstrapped {
                            bootstrapped = true;
                            info!("Client bootstrapped");
                        }
                    }
                    kad::Event::OutboundQueryProgressed {
                        id,
                        result: kad::QueryResult::GetProviders(Ok(
                            kad::GetProvidersOk::FoundProviders { providers: found, .. },
                        )),
                        ..
                    } => {
                        if let Some(&layer) = pending_queries.get(&id) {
                            for p in &found {
                                // Skip peers we already know have failed.
                                if !failed_peers.contains(p) {
                                    providers.insert(layer, *p);
                                    provider_cache.insert(layer, *p);
                                    info!("Client: found provider for layer {}: {:?}", layer, p);
                                    break;
                                }
                            }
                            pending_queries.remove(&id);
                        }
                    }
                    kad::Event::OutboundQueryProgressed {
                        id,
                        result: kad::QueryResult::GetProviders(Ok(
                            kad::GetProvidersOk::FinishedWithNoAdditionalRecord { .. },
                        )),
                        ..
                    } => {
                        if let Some(&layer) = pending_queries.get(&id) {
                            warn!("Client: no providers for layer {}", layer);
                            pending_queries.remove(&id);
                        }
                    }
                    _ => debug!("Client: kad event: {:?}", e),
                }

                if bootstrapped && !queries_sent {
                    queries_sent = true;
                    info!("Client: looking up providers for layers: {:?}", layers);
                    for &layer in &layers {
                        let qid = dht::find_providers_for_layer(swarm.behaviour_mut(), layer);
                        pending_queries.insert(qid, layer);
                    }
                }

                if !chain_built
                    && bootstrapped
                    && pending_queries.is_empty()
                    && !layers.is_empty()
                    && layers.iter().all(|l| providers.contains_key(l))
                {
                    let chain: Vec<PeerId> = layers
                        .iter()
                        .filter_map(|l| providers.get(l).cloned())
                        .collect();

                    if chain.is_empty() {
                        if retry_count >= MAX_RETRIES {
                            return Err(anyhow!("Could not build chain after {} retries", retry_count));
                        }
                        // Retry Kad queries
                        queries_sent = false;
                        continue;
                    }

                    info!("Client: resolved chain: {:?}", chain);
                    let request = ForwardPassRequest::build(
                        &chain,
                        0,
                        1,
                        Dtype::F32,
                        &[1, payload.len() as u64 / 4],
                        &payload,
                        None,
                    )?;

                    let first_peer = chain[0];
                    current_chain = chain;
                    info!("Client: sending forward pass to {:?}", first_peer);
                    swarm.behaviour_mut().request_response.send_request(&first_peer, request);
                    chain_built = true;
                }
            }
            SwarmEvent::Behaviour(InferenceBehaviourEvent::RequestResponse(e)) => {
                match e {
                    request_response::Event::Message {
                        message:
                            request_response::Message::Response { response, .. },
                        ..
                    } => {
                        let rid = response.request_id().unwrap_or(0);
                        info!("Client: received response for request_id {}", rid);
                        result = Some(response);
                    }
                    request_response::Event::OutboundFailure { peer, error, .. } => {
                        error!("Client: outbound request failed to {:?}: {:?}", peer, error);
                        if retry_count >= MAX_RETRIES {
                            return Err(anyhow!("Outbound request failed after {} retries: {:?}", retry_count, error));
                        }
                        retry_count += 1;

                        // Mark the actual failing peer (from the event) as
                        // failed and evict it from all cached layers.
                        let failed = peer;
                        failed_peers.insert(failed);
                        for &layer in &layers {
                            provider_cache.evict(layer, &failed);
                        }

                        // Clear resolved providers so retry re-queries the DHT,
                        // skipping failed peers.
                        providers.clear();
                        queries_sent = false;
                        chain_built = false;
                        current_chain.clear();
                        info!("Client: failover retry {}/{}", retry_count, MAX_RETRIES);
                    }
                    _ => debug!("Client: rr event: {:?}", e),
                }
            }
            SwarmEvent::ConnectionClosed { peer_id, .. } => {
                warn!("Client: connection closed with {:?}", peer_id);
            }
            SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                warn!("Client: outgoing connection error to {:?}: {:?}", peer_id, error);
            }
            e => debug!("Client event: {:?}", e),
        }

        if result.is_some() {
            return Ok(result.take().unwrap());
        }
    }
}

fn handle_kad_event(event: kad::Event) {
    match event {
        kad::Event::RoutingUpdated { peer, is_new_peer, .. } => {
            info!("Kademlia routing updated: peer={:?}, new={}", peer, is_new_peer);
        }
        kad::Event::OutboundQueryProgressed { id, result, .. } => {
            debug!("Kademlia query {:?} result: {:?}", id, result);
        }
        _ => debug!("Kademlia event: {:?}", event),
    }
}
