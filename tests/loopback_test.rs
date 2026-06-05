use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use libp2p::Multiaddr;
use rand::RngCore;
use tokio::sync::RwLock;
use tracing::info;

use inferencemesh_compute::ComputeEngine;

#[tokio::test]
async fn test_loopback_single_worker() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();

    let bootstrap_port = 19000u16;
    let bootstrap_key = libp2p::identity::Keypair::generate_ed25519();
    let mut bootstrap_swarm = inferencemesh_p2p::swarm::build_swarm(&bootstrap_key)?;
    let tcp_addr: Multiaddr = format!("/ip4/127.0.0.1/tcp/{}", bootstrap_port).parse()?;
    bootstrap_swarm.listen_on(tcp_addr.clone())?;

    tokio::spawn(async move {
        let _ = inferencemesh_p2p::swarm::run_bootstrap(&mut bootstrap_swarm, bootstrap_port).await;
    });

    let worker_address: Arc<RwLock<Option<Multiaddr>>> = Arc::new(RwLock::new(None));
    let worker_id_store: Arc<RwLock<Option<libp2p::PeerId>>> = Arc::new(RwLock::new(None));

    let wa_addr = worker_address.clone();
    let wa_id = worker_id_store.clone();
    let bs_a = tcp_addr.clone();

    let worker_key = libp2p::identity::Keypair::generate_ed25519();
    *wa_id.write().await = Some(worker_key.public().to_peer_id());

    tokio::spawn(async move {
        let mut swarm = inferencemesh_p2p::swarm::build_swarm(&worker_key).unwrap();
        swarm.listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap()).unwrap();
        let mut got_addr = false;
        loop {
            tokio::select! {
                event = swarm.select_next_some() => {
                    if let libp2p::swarm::SwarmEvent::NewListenAddr { address, .. } = event {
                        if !got_addr { *wa_addr.write().await = Some(address); got_addr = true; }
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(200)) => { if got_addr { break; } }
            }
        }
        let _ = inferencemesh_p2p::swarm::run_worker(
            &mut swarm, vec![bs_a], 1, 1, None, HashMap::new(), None,
        ).await;
    });

    tokio::time::sleep(Duration::from_secs(5)).await;

    let wa = worker_address.read().await.clone().unwrap();
    let wa_peer = worker_id_store.read().await.clone().unwrap();

    let mut client_swarm = inferencemesh_p2p::swarm::build_swarm(
        &libp2p::identity::Keypair::generate_ed25519(),
    )?;
    let mut payload = vec![0u8; 1024 * 1024];
    rand::thread_rng().fill_bytes(&mut payload);
    let expected_len = payload.len();

    let mut worker_addresses = HashMap::new();
    worker_addresses.insert(wa_peer, wa);

    let response = tokio::time::timeout(
        Duration::from_secs(30),
        inferencemesh_p2p::swarm::run_client(
            &mut client_swarm, vec![tcp_addr], vec![1], payload, worker_addresses,
        ),
    )
    .await
    .expect("Client timed out")?;

    assert_eq!(response.tensor_data()?.len(), expected_len);
    info!("Loopback test passed!");
    Ok(())
}

#[tokio::test]
async fn test_loopback_compute_single_worker() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();

    let bootstrap_port = 19001u16;
    let bootstrap_key = libp2p::identity::Keypair::generate_ed25519();
    let mut bootstrap_swarm = inferencemesh_p2p::swarm::build_swarm(&bootstrap_key)?;
    let tcp_addr: Multiaddr = format!("/ip4/127.0.0.1/tcp/{}", bootstrap_port).parse()?;
    bootstrap_swarm.listen_on(tcp_addr.clone())?;

    tokio::spawn(async move {
        let _ = inferencemesh_p2p::swarm::run_bootstrap(&mut bootstrap_swarm, bootstrap_port).await;
    });

    let worker_address: Arc<RwLock<Option<Multiaddr>>> = Arc::new(RwLock::new(None));
    let worker_id_store: Arc<RwLock<Option<libp2p::PeerId>>> = Arc::new(RwLock::new(None));
    let wa_addr = worker_address.clone();
    let wa_id = worker_id_store.clone();
    let bs_a = tcp_addr.clone();

    let worker_key = libp2p::identity::Keypair::generate_ed25519();
    *wa_id.write().await = Some(worker_key.public().to_peer_id());

    let mut engine = ComputeEngine::new(candle_core::Device::Cpu);
    engine.load_test_block(1)?;
    let engine = Arc::new(engine);

    tokio::spawn(async move {
        let mut swarm = inferencemesh_p2p::swarm::build_swarm(&worker_key).unwrap();
        swarm.listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap()).unwrap();
        let mut got_addr = false;
        loop {
            tokio::select! {
                event = swarm.select_next_some() => {
                    if let libp2p::swarm::SwarmEvent::NewListenAddr { address, .. } = event {
                        if !got_addr { *wa_addr.write().await = Some(address); got_addr = true; }
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(200)) => { if got_addr { break; } }
            }
        }
        let _ = inferencemesh_p2p::swarm::run_worker(
            &mut swarm, vec![bs_a], 1, 1, Some(engine), HashMap::new(), None,
        ).await;
    });

    tokio::time::sleep(Duration::from_secs(5)).await;

    let wa = worker_address.read().await.clone().unwrap();
    let wa_peer = worker_id_store.read().await.clone().unwrap();

    let mut client_swarm = inferencemesh_p2p::swarm::build_swarm(
        &libp2p::identity::Keypair::generate_ed25519(),
    )?;

    let tensor_bytes: Vec<u8> = vec![0u8; 1 * 8 * 128 * 4];

    let mut worker_addresses = HashMap::new();
    worker_addresses.insert(wa_peer, wa);

    let _response = tokio::time::timeout(
        Duration::from_secs(30),
        inferencemesh_p2p::swarm::run_client(
            &mut client_swarm, vec![tcp_addr], vec![1], tensor_bytes, worker_addresses,
        ),
    )
    .await
    .expect("Client timed out")?;

    info!("Compute loopback test passed!");
    Ok(())
}
