use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::sync::RwLock;
use tracing::info;

/// Chaos test: worker B is a backup for layer 1. Client sends to worker A,
/// worker A gets killed, client detects failure and retries with worker B.

#[tokio::test]
async fn test_chaos_worker_failover() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();

    let bootstrap_port = 20000u16;
    let bootstrap_key = libp2p::identity::Keypair::generate_ed25519();
    let mut bootstrap_swarm = inferencemesh_p2p::swarm::build_swarm(&bootstrap_key)?;
    let tcp_addr: libp2p::Multiaddr =
        format!("/ip4/127.0.0.1/tcp/{}", bootstrap_port).parse()?;
    bootstrap_swarm.listen_on(tcp_addr.clone())?;

    tokio::spawn(async move {
        let _ =
            inferencemesh_p2p::swarm::run_bootstrap(&mut bootstrap_swarm, bootstrap_port).await;
    });

    // Worker A — primary
    let wa_addr: Arc<RwLock<Option<libp2p::Multiaddr>>> = Arc::new(RwLock::new(None));
    let wa_id: Arc<RwLock<Option<libp2p::PeerId>>> = Arc::new(RwLock::new(None));
    let wa_addr_clone = wa_addr.clone();
    let _wa_id_clone = wa_id.clone();
    let bs_a = tcp_addr.clone();
    let wa_key = libp2p::identity::Keypair::generate_ed25519();
    *wa_id.write().await = Some(wa_key.public().to_peer_id());

    let wa_handle = tokio::spawn(async move {
        let mut swarm =
            inferencemesh_p2p::swarm::build_swarm(&wa_key).unwrap();
        swarm
            .listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap())
            .unwrap();
        let mut got_addr = false;
        loop {
            tokio::select! {
                event = futures::StreamExt::select_next_some(&mut swarm) => {
                    if let libp2p::swarm::SwarmEvent::NewListenAddr { address, .. } = event {
                        if !got_addr { *wa_addr_clone.write().await = Some(address); got_addr = true; }
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(200)) => { if got_addr { break; } }
            }
        }
        let _ = inferencemesh_p2p::swarm::run_worker(
            &mut swarm,
            vec![bs_a],
            1,
            1,
            None,
            HashMap::new(),
            None,
        )
        .await;
    });

    // Worker B — backup
    let wb_addr: Arc<RwLock<Option<libp2p::Multiaddr>>> = Arc::new(RwLock::new(None));
    let wb_id: Arc<RwLock<Option<libp2p::PeerId>>> = Arc::new(RwLock::new(None));
    let wb_addr_clone = wb_addr.clone();
    let _wb_id_clone = wb_id.clone();
    let bs_b = tcp_addr.clone();
    let wb_key = libp2p::identity::Keypair::generate_ed25519();
    *wb_id.write().await = Some(wb_key.public().to_peer_id());

    tokio::spawn(async move {
        let mut swarm =
            inferencemesh_p2p::swarm::build_swarm(&wb_key).unwrap();
        swarm
            .listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap())
            .unwrap();
        let mut got_addr = false;
        loop {
            tokio::select! {
                event = futures::StreamExt::select_next_some(&mut swarm) => {
                    if let libp2p::swarm::SwarmEvent::NewListenAddr { address, .. } = event {
                        if !got_addr { *wb_addr_clone.write().await = Some(address); got_addr = true; }
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(200)) => { if got_addr { break; } }
            }
        }
        let _ = inferencemesh_p2p::swarm::run_worker(
            &mut swarm,
            vec![bs_b],
            1,
            1,
            None,
            HashMap::new(),
            None,
        )
        .await;
    });

    tokio::time::sleep(Duration::from_secs(5)).await;

    let wa = wa_addr.read().await.clone().unwrap();
    let wb = wb_addr.read().await.clone().unwrap();
    let wa_peer = wa_id.read().await.clone().unwrap();
    let wb_peer = wb_id.read().await.clone().unwrap();

    info!("Worker A: {:?} at {}", wa_peer, wa);
    info!("Worker B: {:?} at {}", wb_peer, wb);

    let client_key = libp2p::identity::Keypair::generate_ed25519();
    let mut client_swarm =
        inferencemesh_p2p::swarm::build_swarm(&client_key)?;

    let mut payload = vec![0u8; 1024 * 256];
    rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut payload);

    let mut worker_addresses = HashMap::new();
    worker_addresses.insert(wa_peer, wa);
    worker_addresses.insert(wb_peer, wb);

    // Kill worker A after a short delay — the first request will fail, retry picks B
    let wa_handle_clone = wa_handle;
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(500)).await;
        info!("Killing worker A");
        wa_handle_clone.abort();
    });

    let response = tokio::time::timeout(
        Duration::from_secs(30),
        inferencemesh_p2p::swarm::run_client(
            &mut client_swarm,
            vec![tcp_addr],
            vec![1],
            payload.clone(),
            worker_addresses,
        ),
    )
    .await
    .expect("Client timed out")?;

    let data = response.tensor_data()?;
    assert!(!data.is_empty(), "Response data should not be empty");
    info!("Chaos failover test passed! Response length: {}", data.len());
    Ok(())
}
