import torch
import torch.nn as nn
import hivemind
from hivemind.moe.server.layers.custom_experts import register_expert_class

# Define and register your custom expert
class CustomFiveLayerNN(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        )
    
    def forward(self, x):
        return self.layers(x)

custom_nn_sample_input = lambda batch_size, hid_dim: torch.empty((batch_size, hid_dim))
register_expert_class("custom_5layer", custom_nn_sample_input)(CustomFiveLayerNN)

# Create server with explicit NAT traversal options
server = hivemind.moe.Server.create(
    expert_cls="custom_5layer",
    hidden_dim=512,
    num_experts=1,
    expert_pattern="expert.[11:12]",
    initial_peers=["/p2p/12D3KooWHUZDTHPYksAA1PgTcmza2jcZJCyaU2j6us2yajEpBZH9"],
    use_ipfs = True,
    start=True
)

print(f"Server running at: {server.dht.get_visible_maddrs()}")