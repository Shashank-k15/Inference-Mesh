use inferencemesh_protocol::{
    ForwardPassCodec, ForwardPassProtocol, ForwardPassRequest, ForwardPassResponse,
};
use libp2p::autonat;
use libp2p::dcutr;
use libp2p::identify;
use libp2p::kad::{self, Mode};
use libp2p::relay;
use libp2p::request_response;
use libp2p::swarm::NetworkBehaviour;

#[derive(NetworkBehaviour)]
#[behaviour(to_swarm = "InferenceBehaviourEvent")]
pub struct InferenceBehaviour {
    pub kademlia: kad::Behaviour<kad::store::MemoryStore>,
    pub request_response: request_response::Behaviour<ForwardPassCodec>,
    pub identify: identify::Behaviour,
    pub autonat: autonat::Behaviour,
    pub relay: relay::Behaviour,
    pub dcutr: dcutr::Behaviour,
}

#[allow(clippy::large_enum_variant, dead_code)]
#[derive(Debug)]
pub enum InferenceBehaviourEvent {
    Kademlia(kad::Event),
    RequestResponse(
        request_response::Event<ForwardPassRequest, ForwardPassResponse>,
    ),
    Identify(identify::Event),
    Autonat(autonat::Event),
    Relay(relay::Event),
    Dcutr(dcutr::Event),
}

impl From<kad::Event> for InferenceBehaviourEvent {
    fn from(e: kad::Event) -> Self {
        InferenceBehaviourEvent::Kademlia(e)
    }
}

impl From<request_response::Event<ForwardPassRequest, ForwardPassResponse>>
    for InferenceBehaviourEvent
{
    fn from(e: request_response::Event<ForwardPassRequest, ForwardPassResponse>) -> Self {
        InferenceBehaviourEvent::RequestResponse(e)
    }
}

impl From<identify::Event> for InferenceBehaviourEvent {
    fn from(e: identify::Event) -> Self {
        InferenceBehaviourEvent::Identify(e)
    }
}

impl From<autonat::Event> for InferenceBehaviourEvent {
    fn from(e: autonat::Event) -> Self {
        InferenceBehaviourEvent::Autonat(e)
    }
}

impl From<relay::Event> for InferenceBehaviourEvent {
    fn from(e: relay::Event) -> Self {
        InferenceBehaviourEvent::Relay(e)
    }
}

impl From<dcutr::Event> for InferenceBehaviourEvent {
    fn from(e: dcutr::Event) -> Self {
        InferenceBehaviourEvent::Dcutr(e)
    }
}

impl InferenceBehaviour {
    pub fn new(keypair: &libp2p::identity::Keypair) -> Self {
        let peer_id = keypair.public().to_peer_id();

        let mut kademlia = kad::Behaviour::new(peer_id, kad::store::MemoryStore::new(peer_id));
        kademlia.set_mode(Some(Mode::Server));

        let request_response = request_response::Behaviour::new(
            [(ForwardPassProtocol, request_response::ProtocolSupport::Full)],
            request_response::Config::default(),
        );

        let identify = identify::Behaviour::new(
            identify::Config::new("/inferencemesh/1.0.0".to_string(), keypair.public())
                .with_agent_version("inferencemesh/0.1.0".to_string()),
        );

        let autonat = autonat::Behaviour::new(peer_id, autonat::Config::default());

        let relay = relay::Behaviour::new(peer_id, relay::Config::default());

        let dcutr = dcutr::Behaviour::new(peer_id);

        Self {
            kademlia,
            request_response,
            identify,
            autonat,
            relay,
            dcutr,
        }
    }
}
