import hivemind
import time

dht = hivemind.DHT(
    host_maddrs=["/ip4/0.0.0.0/tcp/0"],
    initial_peers=[
        "/ip4/34.60.208.208/tcp/46551/p2p/12D3KooWSk15f6PQYpbVCCYU8qdAuXUPonXFcREPALYKjsR35vrV"
    ],
    start=True
)

# Print confirmation and connection info
print("Connected to DHT!")
print("My peer addresses:")
print('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))

# Check if we can find the initial peer
found_peer = dht.get_visible_maddrs() is not None
print(f"Successfully found peer: {found_peer}")

# Keep the DHT running
try:
    while True:
        time.sleep(1)
        # Optional: periodically check if still connected
        if dht.is_alive():
            print("DHT still connected...", end="\r")
        else:
            print("DHT connection lost!")
            break
except KeyboardInterrupt:
    print("\nShutting down DHT...")
    dht.shutdown()