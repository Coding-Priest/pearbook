import torch
import hivemind

dht = hivemind.DHT(
    initial_peers=["/p2p/12D3KooWHUZDTHPYksAA1PgTcmza2jcZJCyaU2j6us2yajEpBZH9"],
    use_ipfs = True,
    client_mode=True, start=True)

[expert24] = hivemind.moe.get_experts(dht, ["expert.14"])
print(expert24)

dummy = torch.rand(3, 512)
out = expert24(dummy)

print(out)