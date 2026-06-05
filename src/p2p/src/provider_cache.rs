use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use libp2p::PeerId;

#[derive(Debug, Clone)]
struct ProviderEntry {
    peer_id: PeerId,
    discovered_at: Instant,
}

#[derive(Debug, Default)]
pub struct ProviderCache {
    providers: HashMap<u32, Vec<ProviderEntry>>,
    ttl: Duration,
}

impl ProviderCache {
    pub fn new(ttl: Duration) -> Self {
        Self {
            providers: HashMap::new(),
            ttl,
        }
    }

    pub fn insert(&mut self, layer: u32, peer_id: PeerId) {
        let entry = ProviderEntry {
            peer_id,
            discovered_at: Instant::now(),
        };
        self.providers.entry(layer).or_default().push(entry);
    }

    pub fn get_excluding(&self, layer: u32, exclude: &HashSet<PeerId>) -> Vec<PeerId> {
        let now = Instant::now();
        self.providers
            .get(&layer)
            .map(|entries| {
                entries
                    .iter()
                    .filter(|e| {
                        now.duration_since(e.discovered_at) < self.ttl
                            && !exclude.contains(&e.peer_id)
                    })
                    .map(|e| e.peer_id)
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn build_chain(
        &self,
        layers: &[u32],
        exclude: &HashSet<PeerId>,
    ) -> Option<Vec<PeerId>> {
        let mut chain = Vec::with_capacity(layers.len());
        for &layer in layers {
            let candidates = self.get_excluding(layer, exclude);
            if candidates.is_empty() {
                return None;
            }
            chain.push(candidates[0]);
        }
        Some(chain)
    }

    pub fn evict(&mut self, layer: u32, peer_id: &PeerId) {
        if let Some(entries) = self.providers.get_mut(&layer) {
            entries.retain(|e| &e.peer_id != peer_id);
        }
    }

    pub fn prune_expired(&mut self) {
        let now = Instant::now();
        for entries in self.providers.values_mut() {
            entries.retain(|e| now.duration_since(e.discovered_at) < self.ttl);
        }
        self.providers.retain(|_, v| !v.is_empty());
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.providers.values().map(|v| v.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_insert_and_get() {
        let mut cache = ProviderCache::new(Duration::from_secs(60));
        let peer = PeerId::random();
        cache.insert(1, peer);
        assert_eq!(cache.get_excluding(1, &HashSet::new()), vec![peer]);
    }

    #[test]
    fn test_cache_excludes_failed_peers() {
        let mut cache = ProviderCache::new(Duration::from_secs(60));
        let a = PeerId::random();
        let b = PeerId::random();
        cache.insert(1, a);
        cache.insert(1, b);
        let mut exclude = HashSet::new();
        exclude.insert(a);
        assert_eq!(cache.get_excluding(1, &exclude), vec![b]);
    }

    #[test]
    fn test_cache_evict() {
        let mut cache = ProviderCache::new(Duration::from_secs(60));
        let peer = PeerId::random();
        cache.insert(1, peer);
        cache.evict(1, &peer);
        assert!(cache.get_excluding(1, &HashSet::new()).is_empty());
    }

    #[test]
    fn test_build_chain_returns_none_when_missing_layer() {
        let cache = ProviderCache::new(Duration::from_secs(60));
        assert!(cache
            .build_chain(&[1, 2], &HashSet::new())
            .is_none());
    }

    #[test]
    fn test_build_chain_returns_chain_when_all_layers_present() {
        let mut cache = ProviderCache::new(Duration::from_secs(60));
        let a = PeerId::random();
        let b = PeerId::random();
        cache.insert(1, a);
        cache.insert(2, b);
        let chain = cache.build_chain(&[1, 2], &HashSet::new()).unwrap();
        assert_eq!(chain, vec![a, b]);
    }
}
