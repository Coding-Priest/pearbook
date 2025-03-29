import torch
import torch.nn as nn
import torchvision.models as models
import hivemind
import time

# Configuration
DHT_ADDR = "/ip4/34.60.208.208/tcp/46551/p2p/12D3KooWMuNE7uQczAengmmxXzSRjgoRuer1PqJRHTeY6NrgF5R9"
EXPERT_UID = "vgg16_large.1"

class VGG16FirstHalf(nn.Module):
    """First half of VGG16: features[0] to features[22] (Blocks 1-3)"""
    def __init__(self):
        super().__init__()
        # Load the full VGG16 model
        vgg16 = models.vgg16(pretrained=False)
        
        # Take the first 23 layers (0-22) of the features
        # This includes blocks 1-3 (7 conv layers + 3 max pools)
        self.features = nn.Sequential(*list(vgg16.features.children())[:23])
        
        # Print the architecture
        print("VGG16 First Half Architecture:")
        print(self.features)
        
        # Calculate parameters
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {num_params:,}")
    
    def forward(self, x):
        return self.features(x)

# Connect to the DHT
print(f"Connecting to DHT at {DHT_ADDR}...")
dht = hivemind.DHT(
    host_maddrs=["/ip4/0.0.0.0/tcp/0"],
    initial_peers=[DHT_ADDR],
    start=True
)

print("Connected to DHT!")
print("My peer addresses:")
for addr in dht.get_visible_maddrs():
    print(str(addr))

# Create and initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the first half of VGG16
model = VGG16FirstHalf().to(device)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Define input/output shapes
# Input: [batch_size, 3, 224, 224] - Standard ImageNet size
# Output: [batch_size, 256, 28, 28] - After 3 max pooling layers (224->112->56->28)
input_shape = (None, 3, 224, 224)
output_shape = (None, 256, 28, 28)

# Create a custom ModuleBackend that handles device transfer
class CudaModuleBackend(hivemind.moe.ModuleBackend):
    def forward(self, *inputs):
        # Move inputs to the device before processing
        inputs = [tensor.to(device) for tensor in inputs]
        outputs = super().forward(*inputs)
        return outputs
    
    def backward(self, *grad_outputs):
        # Move gradients to the device before processing
        grad_outputs = [tensor.to(device) for tensor in grad_outputs]
        return super().backward(*grad_outputs)

try:
    # Create the server with a larger batch size for the full dataset
    server = hivemind.moe.Server(
        dht=dht,
        module_backends={
            EXPERT_UID: CudaModuleBackend(
                name=EXPERT_UID,
                module=model,
                optimizer=optimizer,
                args_schema=(hivemind.TensorDescriptor(input_shape, dtype=torch.float32),),
                outputs_schema=hivemind.TensorDescriptor(output_shape, dtype=torch.float32),
                max_batch_size=32  # Increased for the full dataset run
            )
        },
        num_connection_handlers=4,
        update_period=30,
        start=True
    )

    print(f"Server is running! Hosting expert: {EXPERT_UID}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Print memory usage
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.1f} MB")

    # Time tracker for the server
    start_time = time.time()
    request_count = 0
    last_print_time = start_time

    # Keep the server running with periodic status updates
    while True:
        time.sleep(5)
        
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Print status every 30 seconds
        if current_time - last_print_time > 30:
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
                print(f"Server running for {elapsed//60:.0f}m {elapsed%60:.0f}s, GPU memory: {mem_allocated:.1f} MB")
            else:
                print(f"Server running for {elapsed//60:.0f}m {elapsed%60:.0f}s")
            last_print_time = current_time
            
        if not server.ready.is_set():
            print("Server connection lost!")
            break
except KeyboardInterrupt:
    print("\nShutting down server...")
    server.shutdown()
    dht.shutdown()
except Exception as e:
    print(f"Error: {str(e)}")
    try:
        if 'server' in locals():
            server.shutdown()
        if 'dht' in locals():
            dht.shutdown()
    except:
        pass