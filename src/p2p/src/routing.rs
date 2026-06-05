use inferencemesh_protocol::ForwardPassRequest;

pub fn build_next_request(
    request: &ForwardPassRequest,
) -> anyhow::Result<(libp2p::PeerId, ForwardPassRequest)> {
    let route = request.route()?;
    let hop_index = request.hop_index()?;
    let next_index = hop_index + 1;
    let next_peer = route[next_index as usize];

    let mask = request.mask_data()?;
    let mask_ref = mask
        .as_ref()
        .map(|(md, ms, md2)| (*md, ms.as_slice(), md2.as_slice()));

    let next = ForwardPassRequest::build(
        &route,
        next_index,
        request.request_id()?,
        request.tensor_dtype()?,
        &request.tensor_shape()?,
        &request.tensor_data()?,
        mask_ref,
    )?;

    Ok((next_peer, next))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_terminal() {
        use libp2p::PeerId;
        let peer_a = PeerId::random();
        let peer_b = PeerId::random();
        let route = vec![peer_a, peer_b];

        let req = ForwardPassRequest::build(
            &route,
            0,
            42,
            inferencemesh_protocol::Dtype::F32,
            &[1, 4],
            &[0u8; 16],
            None,
        )
        .unwrap();
        assert!(!req.is_terminal().unwrap());

        let req2 = ForwardPassRequest::build(
            &route,
            1,
            42,
            inferencemesh_protocol::Dtype::F32,
            &[1, 4],
            &[0u8; 16],
            None,
        )
        .unwrap();
        assert!(req2.is_terminal().unwrap());
    }

    #[test]
    fn test_build_next_request_increments_hop() {
        use libp2p::PeerId;
        let peer_a = PeerId::random();
        let peer_b = PeerId::random();
        let route = vec![peer_a, peer_b];

        let req = ForwardPassRequest::build(
            &route,
            0,
            42,
            inferencemesh_protocol::Dtype::F32,
            &[1, 4],
            &[0u8; 16],
            None,
        )
        .unwrap();
        assert_eq!(req.hop_index().unwrap(), 0);
        assert_eq!(req.request_id().unwrap(), 42);
        assert!(!req.is_terminal().unwrap());

        let (next_peer, next_req) = build_next_request(&req).unwrap();
        assert_eq!(next_peer, peer_b);
        assert_eq!(next_req.hop_index().unwrap(), 1);
        assert_eq!(next_req.request_id().unwrap(), 42);
        assert!(next_req.is_terminal().unwrap());
    }
}
