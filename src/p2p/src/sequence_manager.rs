use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use libp2p::PeerId;
use tracing::debug;

use crate::types::{RemoteModuleInfo, RemoteSpanInfo, ServerInfo, ServerState};

pub fn compute_spans(
    module_infos: &[RemoteModuleInfo],
    min_state: ServerState,
) -> HashMap<PeerId, RemoteSpanInfo> {
    if module_infos.is_empty() {
        return HashMap::new();
    }

    let mut spans: HashMap<PeerId, RemoteSpanInfo> = HashMap::new();

    for (block_idx, module_info) in module_infos.iter().enumerate() {
        let mut sorted_servers: Vec<(&PeerId, &ServerInfo)> =
            module_info.servers.iter().collect();
        sorted_servers.sort_by_key(|(_, info)| info.state as u8);

        for (peer_id, server_info) in &sorted_servers {
            if (server_info.state as u8) < (min_state as u8) {
                continue;
            }

            let entry = spans.entry(**peer_id).or_insert_with(|| RemoteSpanInfo {
                peer_id: **peer_id,
                start: block_idx,
                end: block_idx + 1,
                server_info: (*server_info).clone(),
            });

            if (entry.server_info.state as u8) < (server_info.state as u8) {
                entry.start = block_idx;
                entry.end = block_idx + 1;
                entry.server_info = (*server_info).clone();
            } else if entry.server_info.state == server_info.state {
                entry.end = block_idx + 1;
            }

            if let (Some(sb), Some(eb)) = (server_info.start_block, server_info.end_block) {
                let offset = module_infos.first()
                    .and_then(|m| m.uid.split('.').last())
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                entry.start = entry.start.max(sb.saturating_sub(offset as u32) as usize);
                entry.end = entry.end.min(
                    (eb.saturating_sub(offset as u32) as usize).min(module_infos.len()),
                );
            }
        }
    }

    spans
}

pub fn spans_by_priority(module_infos: &[RemoteModuleInfo]) -> Vec<RemoteSpanInfo> {
    let spans = compute_spans(module_infos, ServerState::Online);
    let mut vec: Vec<RemoteSpanInfo> = spans.into_values().collect();
    vec.sort_by(|a, b| b.length().cmp(&a.length()));
    vec
}

pub fn spans_containing_block(
    spans: &[RemoteSpanInfo],
    num_blocks: usize,
) -> Vec<Vec<RemoteSpanInfo>> {
    let mut result: Vec<Vec<RemoteSpanInfo>> = vec![Vec::new(); num_blocks];
    for span in spans {
        for block_idx in span.start..span.end {
            if block_idx < num_blocks {
                result[block_idx].push(span.clone());
            }
        }
    }
    result
}

#[derive(Copy, Clone)]
struct State {
    cost_ns: i64,
    node: usize,
}

fn cost_to_nanos(cost: f64) -> i64 {
    (cost * 1_000_000_000.0) as i64
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost_ns.cmp(&self.cost_ns).then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.cost_ns == other.cost_ns && self.node == other.node
    }
}

impl Eq for State {}

pub fn dijkstra_min_latency(
    spans: &[RemoteSpanInfo],
    num_blocks: usize,
    client_server_rtts: &HashMap<PeerId, f64>,
    server_server_rtts: &HashMap<(PeerId, PeerId), f64>,
) -> Option<Vec<RemoteSpanInfo>> {
    let containing = spans_containing_block(spans, num_blocks);

    if containing.is_empty() || containing[0].is_empty() || containing[num_blocks - 1].is_empty() {
        return None;
    }

    let mut node_ids: HashMap<(PeerId, usize), usize> = HashMap::new();
    let mut nodes: Vec<(PeerId, usize)> = Vec::new();
    let start_node = nodes.len();
    nodes.push((PeerId::random(), 0));
    let end_node = nodes.len();
    nodes.push((PeerId::random(), num_blocks));

    for (block_idx, spans_at_block) in containing.iter().enumerate() {
        for span in spans_at_block {
            let key = (span.peer_id, block_idx);
            if !node_ids.contains_key(&key) {
                node_ids.insert(key, nodes.len());
                nodes.push(key);
            }
        }
    }

    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nodes.len()];

    let overhead_delay = 0.018;
    let default_inference_rps = 300.0;
    let default_delay = 0.15;

    for span in &containing[0] {
        let to = node_ids[&(span.peer_id, 0)];
        let rtt = client_server_rtts.get(&span.peer_id).copied().unwrap_or(default_delay);
        let delay = rtt / 2.0 + overhead_delay;
        adj[start_node].push((to, delay));
    }

    for span in &containing[num_blocks - 1] {
        let from = node_ids[&(span.peer_id, num_blocks - 1)];
        let rtt = client_server_rtts.get(&span.peer_id).copied().unwrap_or(default_delay);
        let delay = rtt / 2.0;
        adj[from].push((end_node, delay));
    }

    for block_idx in 0..num_blocks {
        for span in &containing[block_idx] {
            if block_idx + 1 < span.end {
                let from = node_ids[&(span.peer_id, block_idx)];
                let to = node_ids[&(span.peer_id, block_idx + 1)];
                let inference_rps = span.server_info.inference_rps.unwrap_or(default_inference_rps);
                let delay = 1.0 / inference_rps;
                adj[from].push((to, delay));
            }

            if block_idx + 1 < num_blocks && span.end == block_idx + 1 {
                for next_span in &containing[block_idx + 1] {
                    let from = node_ids[&(span.peer_id, block_idx)];
                    let to = node_ids[&(next_span.peer_id, block_idx + 1)];
                    let rtt = server_server_rtts
                        .get(&(span.peer_id, next_span.peer_id))
                        .copied()
                        .or_else(|| {
                            span.server_info.next_pings.as_ref().and_then(|pings| {
                                pings.get(&next_span.peer_id.to_base58()).copied()
                            })
                        })
                        .unwrap_or(default_delay);
                    let delay = rtt / 2.0 + overhead_delay;
                    adj[from].push((to, delay));
                }
            }
        }
    }

    let mut dist: Vec<f64> = vec![f64::INFINITY; nodes.len()];
    let mut prev: Vec<Option<usize>> = vec![None; nodes.len()];
    dist[start_node] = 0.0;

    let mut heap = BinaryHeap::new();
    heap.push(State { cost_ns: 0, node: start_node });

    while let Some(State { cost_ns: ns, node }) = heap.pop() {
        let cost = ns as f64 / 1_000_000_000.0;
        if node == end_node {
            break;
        }
        if cost > dist[node] {
            continue;
        }

        for &(next, weight) in &adj[node] {
            let next_cost = cost + weight;
            if next_cost < dist[next] {
                dist[next] = next_cost;
                prev[next] = Some(node);
                heap.push(State { cost_ns: cost_to_nanos(next_cost), node: next });
            }
        }
    }

    if dist[end_node] == f64::INFINITY {
        return None;
    }

    let mut path_nodes = Vec::new();
    let mut cur = end_node;
    while let Some(p) = prev[cur] {
        path_nodes.push(cur);
        cur = p;
    }
    path_nodes.reverse();

    let mut result: Vec<RemoteSpanInfo> = Vec::new();
    for &n in &path_nodes {
        if n == end_node {
            break;
        }
        let (peer_id, block_idx) = nodes[n];
        if result.is_empty() || result.last().unwrap().peer_id != peer_id {
            let server_info = containing[block_idx]
                .iter()
                .find(|s| s.peer_id == peer_id)
                .map(|s| s.server_info.clone())
                .unwrap_or_else(|| ServerInfo::new(ServerState::Online, 0.0));
            result.push(RemoteSpanInfo {
                peer_id,
                start: block_idx,
                end: block_idx + 1,
                server_info,
            });
        } else {
            result.last_mut().unwrap().end = block_idx + 1;
        }
    }

    result.retain(|s| s.length() > 0);
    debug!("Dijkstra found path with {} spans", result.len());
    Some(result)
}

pub fn make_sequence_max_throughput(
    spans: &[RemoteSpanInfo],
    num_blocks: usize,
    start_index: usize,
    end_index: usize,
) -> Option<Vec<RemoteSpanInfo>> {
    let containing = spans_containing_block(spans, num_blocks);
    let mut result = Vec::new();
    let mut current_index = start_index;

    while current_index < end_index {
        if current_index >= containing.len() || containing[current_index].is_empty() {
            return None;
        }
        let chosen = &containing[current_index][0];
        let span_end = chosen.end.min(end_index);
        result.push(RemoteSpanInfo {
            peer_id: chosen.peer_id,
            start: current_index,
            end: span_end,
            server_info: chosen.server_info.clone(),
        });
        current_index = span_end;
    }

    Some(result)
}
