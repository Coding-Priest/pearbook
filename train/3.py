import hivemind

dht = hivemind.DHT(
    host_maddrs=["/ip4/0.0.0.0/tcp/9000"],
    start=True
)
print('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))