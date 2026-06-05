use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use rand::RngCore;
use tokio::sync::RwLock;
use tracing::info;

#[tokio::test]
async fn test_concurrent_clients() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();

    let bootstrap_port = 21000u16;
    let bootstrap_key = libp2p::identity::Keypair::generate_ed25519();
    let mut bootstrap_swarm =
        inferencemesh_p2p::swarm::build_swarm(&bootstrap_key)?;
    let tcp_addr: libp2p::Multiaddr =
        format!("/ip4/127.0.0.1/tcp/{}", bootstrap_port).parse()?;
    bootstrap_swarm.listen_on(tcp_addr.clone())?;

    tokio::spawn(async move {
        let _ =
            inferencemesh_p2p::swarm::run_bootstrap(&mut bootstrap_swarm, bootstrap_port).await;
    });

    let worker_addr: Arc<RwLock<Option<libp2p::Multiaddr>>> = Arc::new(RwLock::new(None));
    let worker_id: Arc<RwLock<Option<libp2p::PeerId>>> = Arc::new(RwLock::new(None));
    let wa = worker_addr.clone();
    let _wi = worker_id.clone();
    let bs = tcp_addr.clone();

    let worker_key = libp2p::identity::Keypair::generate_ed25519();
    *worker_id.write().await = Some(worker_key.public().to_peer_id());

    tokio::spawn(async move {
        let mut swarm =
            inferencemesh_p2p::swarm::build_swarm(&worker_key).unwrap();
        swarm
            .listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap())
            .unwrap();
        let mut got_addr = false;
        loop {
            tokio::select! {
                event = futures::StreamExt::select_next_some(&mut swarm) => {
                    if let libp2p::swarm::SwarmEvent::NewListenAddr { address, .. } = event {
                        if !got_addr { *wa.write().await = Some(address); got_addr = true; }
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(200)) => { if got_addr { break; } }
            }
        }
        let _ = inferencemesh_p2p::swarm::run_worker(
            &mut swarm,
            vec![bs],
            1,
            1,
            None,
            HashMap::new(),
            None,
        )
        .await;
    });

    tokio::time::sleep(Duration::from_secs(5)).await;

    let w_addr = worker_addr.read().await.clone().unwrap();
    let w_peer = worker_id.read().await.clone().unwrap();
    info!("Worker: {:?} at {}", w_peer, w_addr);

    let concurrent_clients = 5;
    let mut handles = Vec::new();

    for i in 0..concurrent_clients {
        let bs_clone = tcp_addr.clone();
        let w_addr_clone = w_addr.clone();
        let w_peer_clone = w_peer;

        let handle = tokio::spawn(async move {
            let mut payload = vec![0u8; 1024 * 64];
            rand::thread_rng().fill_bytes(&mut payload);

            let mut worker_addresses = HashMap::new();
            worker_addresses.insert(w_peer_clone, w_addr_clone);

            let client_key = libp2p::identity::Keypair::generate_ed25519();
            let mut client_swarm =
                inferencemesh_p2p::swarm::build_swarm(&client_key).unwrap();

            let result = tokio::time::timeout(
                Duration::from_secs(15),
                inferencemesh_p2p::swarm::run_client(
                    &mut client_swarm,
                    vec![bs_clone],
                    vec![1],
                    payload,
                    worker_addresses,
                ),
            )
            .await;

            match result {
                Ok(Ok(response)) => {
                    let len = response.tensor_data().unwrap_or_default().len();
                    info!("Client {}: received {} bytes", i, len);
                    Ok(len)
                }
                Ok(Err(e)) => {
                    anyhow::bail!("Client {}: error: {:?}", i, e)
                }
                Err(_) => {
                    anyhow::bail!("Client {}: timed out", i)
                }
            }
        });
        handles.push(handle);
    }

    let mut results = Vec::new();
    for handle in handles {
        match handle.await {
            Ok(r) => results.push(r),
            Err(e) => eprintln!("Join error: {:?}", e),
        }
    }

    let success_count = results.iter().filter(|r| r.is_ok()).count();
    assert_eq!(
        success_count,
        concurrent_clients,
        "Expected all {} clients to succeed, got {}",
        concurrent_clients,
        success_count
    );

    info!(
        "Concurrency test passed! {} clients all successful",
        success_count
    );
    Ok(())
}
