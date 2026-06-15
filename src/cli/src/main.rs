use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use clap::{Parser, Subcommand};
use libp2p::Multiaddr;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;
use candle_nn::Module;

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

        #[arg(long, required = true)]
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
    Generate {
        #[arg(long, required = true)]
        bootstrap: String,

        #[arg(long, required = true, value_delimiter = ',')]
        chain: Vec<u32>,

        #[arg(long, required = true)]
        model_path: String,

        #[arg(long, required = true)]
        prompt: String,

        #[arg(long, default_value = "20")]
        max_tokens: usize,
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
                let model_path = Path::new(mp);
                let config_path = if model_path.is_dir() {
                    model_path.join("config.json")
                } else {
                    model_path.parent().unwrap_or(Path::new(".")).join("config.json")
                };
                let config = inferencemesh_compute::ModelConfig::from_file(&config_path)
                    .unwrap_or_else(|_| {
                        info!("No config.json found, using test config");
                        inferencemesh_compute::ModelConfig {
                            hidden_size: 128,
                            intermediate_size: 256,
                            num_attention_heads: 4,
                            num_key_value_heads: Some(4),
                            num_hidden_layers: 2,
                            rms_norm_eps: 1e-5,
                            rope_theta: 10000.0,
                            max_position_embeddings: Some(2048),
                            max_position_embeddings_fallback: 2048,
                            model_type: Some("llama".into()),
                        }
                    });
                let mut eng =
                    inferencemesh_compute::ComputeEngine::new(candle_device);
                eng.load_blocks(model_path, &config, start, end)?;
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
        Command::Generate {
            bootstrap,
            chain,
            model_path,
            prompt,
            max_tokens,
        } => {
            let bootstrap_addrs: Vec<Multiaddr> = bootstrap
                .split(',')
                .map(|s| s.parse())
                .collect::<Result<Vec<_>, _>>()?;

            let keypair = libp2p::identity::Keypair::generate_ed25519();
            let peer_id = keypair.public().to_peer_id();
            info!("Generate client peer ID: {:?}, chain: {:?}", peer_id, chain);

            let model_path = Path::new(&model_path);
            let tokenizer_path = if model_path.is_dir() {
                model_path.join("tokenizer.json")
            } else {
                model_path.parent().unwrap_or(Path::new(".")).join("tokenizer.json")
            };

            let safetensors_path = if model_path.is_dir() {
                let mut path = model_path.join("model.safetensors");
                if !path.exists() {
                    if let Ok(entries) = std::fs::read_dir(model_path) {
                        for entry in entries.flatten() {
                            let n = entry.file_name();
                            let n = n.to_string_lossy();
                            if n.ends_with(".safetensors") && !n.contains("index.json") {
                                path = entry.path();
                                break;
                            }
                        }
                    }
                }
                path
            } else {
                model_path.to_path_buf()
            };

            let config_path = if model_path.is_dir() {
                model_path.join("config.json")
            } else {
                model_path.parent().unwrap_or(Path::new(".")).join("config.json")
            };

            let config = inferencemesh_compute::ModelConfig::from_file(&config_path)?;
            let device = candle_core::Device::Cpu;
            
            info!("Loading tokenizer...");
            let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;
            
            info!("Loading embeddings and LM head...");
            let data = std::fs::read(&safetensors_path)?;
            let st = safetensors::SafeTensors::deserialize(&data).map_err(|e| anyhow::anyhow!("{:?}", e))?;
            let tensor_view = st.tensor("model.embed_tokens.weight").map_err(|e| anyhow::anyhow!("{:?}", e))?;
            let vocab_size = tensor_view.shape()[0];
            
            let vb = candle_nn::VarBuilder::from_buffered_safetensors(data, candle_core::DType::F32, &device)?;
            
            let embed_weight = vb.get((vocab_size, config.hidden_size), "model.embed_tokens.weight")?;

            let embed_tokens = candle_nn::Embedding::new(embed_weight, config.hidden_size);

            let norm_eps = config.rms_norm_eps as f64;
            let norm = candle_nn::rms_norm(config.hidden_size, norm_eps, vb.pp("model.norm"))?;
            
            let lm_head_weight = vb.get((vocab_size, config.hidden_size), "lm_head.weight")
                .or_else(|_| vb.get((vocab_size, config.hidden_size), "model.embed_tokens.weight"))?;
            
            let mut tokens = tokenizer.encode(prompt.clone(), true).map_err(|e| anyhow::anyhow!(e))?.get_ids().to_vec();
            
            info!("Prompt encoded: {:?}", tokens);

            let mut swarm = inferencemesh_p2p::swarm::build_swarm(&keypair)?;
            let (req_tx, req_rx) = tokio::sync::mpsc::channel(1);
            let (res_tx, mut res_rx) = tokio::sync::mpsc::channel(1);

            let swarm_task = tokio::spawn(async move {
                let res = inferencemesh_p2p::swarm::run_stream_client(
                    &mut swarm,
                    bootstrap_addrs,
                    chain,
                    req_rx,
                    res_tx,
                    HashMap::new(),
                )
                .await;
                if let Err(e) = res {
                    error!("Stream client failed: {:?}", e);
                }
            });

            print!("{}", prompt);
            use std::io::Write;
            std::io::stdout().flush()?;

            for _ in 0..max_tokens {
                let seq_len = tokens.len();
                let token_tensor = candle_core::Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
                
                let embeddings = embed_tokens.forward(&token_tensor)?;
                let b_sz = 1;
                let hidden_size = config.hidden_size;
                let shape = vec![b_sz as u64, seq_len as u64, hidden_size as u64];
                
                let emb_flat = embeddings.flatten_all()?;
                let payload = emb_flat.to_vec1::<f32>()?.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();

                req_tx.send((payload, shape)).await?;
                
                match res_rx.recv().await {
                    Some(Ok(response)) => {
                        let out_data = response.tensor_data()?;
                        let out_f32: Vec<f32> = out_data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                        let out_tensor = candle_core::Tensor::from_vec(out_f32, (1, seq_len, hidden_size), &device)?;
                        
                        let last_hidden = out_tensor.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
                        let normed = candle_nn::ops::rms_norm(&last_hidden, norm.weight(), config.rms_norm_eps as f32)?;
                        let logits = normed.broadcast_matmul(&lm_head_weight.t()?)?;
                        
                        let next_token = logits.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?;
                        tokens.push(next_token);
                        
                        let text = tokenizer.decode(&[next_token], true).map_err(|e| anyhow::anyhow!(e))?;
                        print!("{}", text);
                        std::io::stdout().flush()?;
                    }
                    Some(Err(e)) => {
                        error!("Pipeline error: {:?}", e);
                        break;
                    }
                    None => {
                        error!("Pipeline closed");
                        break;
                    }
                }
            }
            println!();

            swarm_task.abort();
            Ok(())
        }
    }
}
