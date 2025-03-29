import torch
import torch.nn as nn
import hivemind
from hivemind.moe.server.layers.custom_experts import register_expert_class

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

server = hivemind.moe.Server.create(
    expert_cls="custom_5layer",     
    hidden_dim=512,                 
    num_experts=1,          
    expert_pattern="expert.[100:101]",
    initial_peers=["/ip4/34.60.208.208/tcp/46551/p2p/12D3KooWLWcbmR8mhNvrsGnfZtfLUYpuF4mzuT4jq4Wk9YtDGu99"], 
    host_maddrs=["/ip4/0.0.0.0/tcp/0"],
    start=True,
    use_ipfs=True
)

import time
while True:
    time.sleep(1)