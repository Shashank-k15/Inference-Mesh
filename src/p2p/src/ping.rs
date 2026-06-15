use std::collections::HashMap;
use std::time::{Duration, Instant};

use libp2p::PeerId;
use tracing::debug;

#[derive(Debug, Default)]
pub struct PingAggregator {
    rtts: HashMap<PeerId, f64>,
    last_ping: HashMap<PeerId, Instant>,
}

impl PingAggregator {
    pub fn new() -> Self {
        PingAggregator {
            rtts: HashMap::new(),
            last_ping: HashMap::new(),
        }
    }

    pub fn record_rtt(&mut self, peer_id: PeerId, rtt_ms: f64) {
        self.rtts.insert(peer_id, rtt_ms);
        self.last_ping.insert(peer_id, Instant::now());
        debug!("Ping to {:?}: {:.1}ms", peer_id, rtt_ms);
    }

    pub fn get_rtt(&self, peer_id: &PeerId) -> Option<f64> {
        self.rtts.get(peer_id).copied()
    }

    pub fn to_dict(&self) -> HashMap<String, f64> {
        self.rtts
            .iter()
            .map(|(p, r)| (p.to_base58(), *r))
            .collect()
    }

    pub fn stale(&self, peer_id: &PeerId, max_age: Duration) -> bool {
        match self.last_ping.get(peer_id) {
            Some(t) => t.elapsed() > max_age,
            None => true,
        }
    }

    pub fn retain_recent(&mut self, max_age: Duration) {
        let stale_keys: Vec<PeerId> = self
            .rtts
            .keys()
            .filter(|p| self.stale(p, max_age))
            .cloned()
            .collect();
        for key in stale_keys {
            self.rtts.remove(&key);
            self.last_ping.remove(&key);
        }
    }
}
