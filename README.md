# InferenceMesh

A decentralized peer-to-peer pipeline-parallel inference network written in Rust. Split large language models across machines and run inference collaboratively — BitTorrent-style.

## Architecture

```
                    Kademlia DHT (bootstrap node)
                         │
            ┌────────────┼────────────┐
            │            │            │
       [Worker A]   [Worker B]   [Worker C]
      Layers 0-7    Layers 8-15   Layers 16-23
            │            │            │
            └────────────┴────────────┘
                         │
                      [Client]
              (embeddings + LM head)
```

Each worker downloads a subset of transformer blocks, announces them on the DHT, and processes hidden states piped through the chain over QUIC or TCP.

## Quick Start

### 1. Build

```bash
git clone git@github.com:Shashank-k15/Inference-Mesh.git
cd Inference-Mesh
cargo build --release
```

The binary is at `target/release/inferencemesh`.

### 2. Start a bootstrap node

The bootstrap node is the rendezvous point. Start one on a machine with a public IP:

```bash
inferencemesh bootstrap --port 9000
```

Note the peer ID printed on startup — workers and clients will need the bootstrap address.

### 3. Download a model

InferenceMesh uses HuggingFace-format models with safetensors weights. Download any Llama-architecture model:

```bash
# Option A: Use huggingface-cli
pip install huggingface_hub
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir ./models/llama-3.2-1b

# Option B: Clone via git (requires token for gated models)
git clone https://huggingface.co/meta-llama/Llama-3.2-1B ./models/llama-3.2-1b
```

The directory must contain `config.json` and `model.safetensors` (or a safetensors index plus shards).

> **Supported architectures**: Any model using the standard Llama layout (RMSNorm, RoPE, GQA, SwiGLU MLP). This includes Llama 3/3.1/3.2, Mistral, Qwen, and Phi — anything with `model.layers.{N}.self_attn.q_proj` weight naming.

### 4. Launch workers

Run one worker per GPU or machine. Each worker loads a subset of layers determined by `--layers`:
Here is a possible configuration.

**Machine A (layers 0–7):**

```bash
inferencemesh worker \
  --bootstrap /ip4/192.168.1.10/tcp/9000 \
  --layers 0-7 \
  --model-path ./models/llama-3.2-1b \
  --device cuda \
  --port 9100
```

**Machine B (layers 8–15):**

```bash
inferencemesh worker \
  --bootstrap /ip4/192.168.1.10/tcp/9000 \
  --layers 8-15 \
  --model-path ./models/llama-3.2-1b \
  --device cuda \
  --port 9200
```

**Machine C (layers 16–23):**

```bash
inferencemesh worker \
  --bootstrap /ip4/192.168.1.10/tcp/9000 \
  --layers 16-23 \
  --model-path ./models/llama-3.2-1b \
  --device cuda \
  --port 9300
```

> **How many blocks to host?** The system auto-selects based on available GPU memory. Omit `--layers` to fill the GPU automatically, or specify a range to control it manually. A Llama 3.2 1B model has 16 hidden layers — split across 3 machines at ~5–6 layers each.

> **CPU-only fallback:** Omit `--device cuda` to run on CPU. Useful for testing but slow for real inference.

> **Echo mode (no model):** Omit `--model-path` to run in echo mode — the worker mirrors hidden states back untouched. Perfect for network testing without downloading models.

### 5. Check the swarm

Once workers announce their layers, the swarm is live. Each worker will log:

```
Worker announcing layers 0-7
Worker bootstrapped
```

### 6. Submit test inference

```bash
inferencemesh client \
  --bootstrap /ip4/127.0.0.1/tcp/9000 \
  --chain 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
  --model-path ./Models \
  --prompt "repeat a 5 times" \
  --max-tokens 20
```

The client discovers which workers host each layer via the DHT, builds a chain, and sends a test payload through the pipeline. You'll see:

```
Client: found provider for layer 0: PeerId(...)
Client: resolved chain: [PeerId(...), PeerId(...), PeerId(...)]
Client: sending forward pass to PeerId(...)
Client: received response for request_id 1
```
