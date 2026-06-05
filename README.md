# InferenceMesh

A decentralized, peer-to-peer pipeline-parallel inference network written in Rust. Inspired by [Petals](https://github.com/bigscience-workshop/petals), optimized for low-latency systems-level performance, zero-copy memory operations, and robust connection pooling.

## Architecture

InferenceMesh splits large language models (LLMs) across a heterogeneous network of nodes. Each node hosts a subset of transformer blocks and participates in a pipeline-parallel forward pass:

```
[Client] → (Embeddings) → [Node A: Layers 1-10] → [Node B: Layers 11-20] → [Node C: Layers 21-30] → [Client] (LM Head)
```

### Core Flow

1. **Client** tokenizes input, runs the embedding layer locally, and produces hidden states.
2. **Route Resolution** via Kademlia DHT — discover which peers host the required layers.
3. **Pipeline Forward Pass** — hidden states are piped through the chain over QUIC connections.
4. **Client** receives final hidden states, runs the LM head, and samples the next token.

## Tech Stack

| Component        | Crate                       | Purpose                                                    |
| ---------------- | --------------------------- | ---------------------------------------------------------- |
| P2P Networking   | `rust-libp2p`               | Identity, NAT traversal, Kademlia DHT, encrypted transport |
| Data Transport   | QUIC (via libp2p)           | Multiplexed, low-latency UDP streams                       |
| Serialization    | `capnp` (Cap'n Proto)       | Zero-copy tensor serialization                             |
| Inference Engine | `candle-core` / `candle-nn` | Pure-Rust ML framework with CUDA/Metal/CPU backends        |
| Async Runtime    | `tokio`                     | Multi-threaded async executor                              |

## Project Structure

```
src/
├── protocol/   # Cap'n Proto schemas, codec, request/response types
├── compute/    # Candle-based transformer block execution engine
├── p2p/        # libp2p swarm, DHT, routing, provider cache
└── cli/        # Binary entry point (bootstrap, worker, client modes)

tests/          # Integration tests (loopback, chaos, concurrency, math parity, memory)
docs/           # Architecture documentation and specs
examples/       # Usage examples
```

## Building

```bash
cargo build --workspace
```

## Running

### Bootstrap Node

```bash
cargo run -- bootstrap --port 9000
```

### Worker Node

```bash
cargo run -- worker --bootstrap /ip4/BOOTSTRAP_IP/tcp/9000 --layers 1-10 --port 9001
```

With model weights:

```bash
cargo run -- worker --bootstrap /ip4/BOOTSTRAP_IP/tcp/9000 --layers 1-10 --model-path ./model/ --device cuda --port 9001
```

### Client

```bash
cargo run -- client --bootstrap /ip4/BOOTSTRAP_IP/tcp/9000 --chain 1,2,3,4,5
```

## Testing

```bash
# All tests
cargo test --workspace

# Specific integration tests
cargo test --test loopback_test
cargo test --test math_parity_test
cargo test --test memory_test
cargo test --test chaos_test
cargo test --test concurrency_test
```
