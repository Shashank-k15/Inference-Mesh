use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use clap::{Parser, Subcommand};
use libp2p::Multiaddr;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "inferencemesh", version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Bootstrap {
        #[arg(long, default_value = "9000")]
        port: u16,
    },
    Worker {
        #[arg(long, required = true)]
        bootstrap: String,

        #[arg(long, default_value = "1", value_delimiter = '-')]
        layers: String,

        #[arg(long)]
        model_path: Option<String>,

        #[arg(long, default_value = "cpu")]
        device: String,

        /// Port to listen on (0 = OS-assigned ephemeral port).
        #[arg(long, default_value = "0")]
        port: u16,
    },
    Client {
        #[arg(long, required = true)]
        bootstrap: String,

        #[arg(long, required = true, value_delimiter = ',')]
        chain: Vec<u32>,
    },
}

fn parse_layers(s: &str) -> Result<(u32, u32)> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 2 {
        return Err(anyhow::anyhow!(
            "layers must be in format 'start-end', e.g. '1-10'"
        ));
    }
    let start: u32 = parts[0].parse()?;
    let end: u32 = parts[1].parse()?;
    if start > end {
        return Err(anyhow::anyhow!("start layer must be <= end layer"));
    }
    Ok((start, end))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Bootstrap { port } => {
            let keypair = libp2p::identity::Keypair::generate_ed25519();
            let peer_id = keypair.public().to_peer_id();
            info!("Bootstrap node peer ID: {:?}", peer_id);

            let mut swarm = inferencemesh_p2p::swarm::build_swarm(&keypair)?;
            inferencemesh_p2p::swarm::run_bootstrap(&mut swarm, port).await
        }
        Command::Worker {
            bootstrap,
            layers,
            model_path,
            device,
            port,
        } => {
            let (start, end) = parse_layers(&layers)?;
            let bootstrap_addrs: Vec<Multiaddr> = bootstrap
                .split(',')
                .map(|s| s.parse())
                .collect::<Result<Vec<_>, _>>()?;

            let keypair = libp2p::identity::Keypair::generate_ed25519();
            let peer_id = keypair.public().to_peer_id();
            info!("Worker peer ID: {:?}, layers: {}-{}", peer_id, start, end);

            let engine = if let Some(ref mp) = model_path {
                let candle_device = match device.as_str() {
                    "cuda" | "cuda:0" => {
                        candle_core::Device::new_cuda(0)
                            .expect("CUDA device not available")
                    }
                    "metal" => {
                        candle_core::Device::new_metal(0)
                            .expect("Metal device not available")
                    }
                    _ => candle_core::Device::Cpu,
                };
                let mut eng =
                    inferencemesh_compute::ComputeEngine::new(candle_device);
                let model_path = Path::new(mp);
                eng.load_blocks(model_path, start, end)?;
                info!("Loaded layers {}-{} from {}", start, end, mp);
                Some(std::sync::Arc::new(eng))
            } else {
                info!("No model path specified, running in echo mode");
                None
            };

            let mut swarm = inferencemesh_p2p::swarm::build_swarm(&keypair)?;

            // Listen on TCP and QUIC so peers can reach this worker.
            let tcp_addr: Multiaddr = format!("/ip4/0.0.0.0/tcp/{}", port).parse()?;
            swarm.listen_on(tcp_addr)?;
            let quic_addr: Multiaddr = format!("/ip4/0.0.0.0/udp/{}/quic-v1", port).parse()?;
            swarm.listen_on(quic_addr)?;

            inferencemesh_p2p::swarm::run_worker(
                &mut swarm,
                bootstrap_addrs,
                start,
                end,
                engine,
                HashMap::new(),
                None,
            )
            .await
        }
        Command::Client { bootstrap, chain } => {
            let bootstrap_addrs: Vec<Multiaddr> = bootstrap
                .split(',')
                .map(|s| s.parse())
                .collect::<Result<Vec<_>, _>>()?;

            let keypair = libp2p::identity::Keypair::generate_ed25519();
            let peer_id = keypair.public().to_peer_id();
            info!("Client peer ID: {:?}, chain: {:?}", peer_id, chain);

            let payload = vec![0u8; 10_000_000];
            info!("Client payload size: {} bytes", payload.len());

            let mut swarm = inferencemesh_p2p::swarm::build_swarm(&keypair)?;
            let response = inferencemesh_p2p::swarm::run_client(
                &mut swarm,
                bootstrap_addrs,
                chain,
                payload,
                HashMap::new(),
            )
            .await?;

            let payload_len = response.tensor_data().unwrap_or_default().len();
            info!("Client received response, payload length: {}", payload_len);
            Ok(())
        }
    }
}
