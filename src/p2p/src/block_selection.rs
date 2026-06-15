use crate::types::{RemoteModuleInfo, ServerState};

pub fn choose_best_blocks(
    num_blocks: usize,
    module_infos: &[RemoteModuleInfo],
) -> Option<Vec<usize>> {
    if module_infos.is_empty() || num_blocks == 0 {
        return None;
    }

    let total_blocks = module_infos.len();
    let throughputs = compute_throughputs(module_infos, ServerState::Joining);

    if num_blocks > total_blocks {
        return Some((0..total_blocks).collect());
    }

    let mut best_start = 0;
    let mut best_min = f64::MAX;

    for start in 0..=(total_blocks - num_blocks) {
        let slice = &throughputs[start..start + num_blocks];
        let min_val = slice.iter().cloned().fold(f64::INFINITY, f64::min);
        if min_val < best_min {
            best_min = min_val;
            best_start = start;
        }
    }

    Some((best_start..best_start + num_blocks).collect())
}

pub fn compute_throughputs(
    module_infos: &[RemoteModuleInfo],
    min_state: ServerState,
) -> Vec<f64> {
    let n = module_infos.len();
    let mut result = vec![0.0; n];

    for (i, module_info) in module_infos.iter().enumerate() {
        for server_info in module_info.servers.values() {
            if (server_info.state as u8) >= (min_state as u8) {
                result[i] += server_info.throughput;
            }
        }
    }

    result
}

pub fn should_choose_other_blocks(
    _local_peer_throughput: f64,
    _local_num_blocks: usize,
    module_infos: &[RemoteModuleInfo],
    balance_quality: f64,
) -> bool {
    if module_infos.is_empty() || _local_num_blocks == 0 {
        return false;
    }

    let mut throughputs = compute_throughputs(module_infos, ServerState::Joining);
    let initial_min = throughputs
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);

    if initial_min < 1e-6 {
        let has_disjoint = throughputs.iter().any(|&t| t < 1e-6);
        if has_disjoint {
            return false;
        }
    }

    let needed_min = initial_min / balance_quality;
    let current_min = throughputs
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);

    current_min < needed_min
}
