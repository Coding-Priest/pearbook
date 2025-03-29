import torch
import torch.nn as nn
import torchvision.models as models
import hivemind
import time

# Configuration
DHT_ADDR = "/ip4/34.60.208.208/tcp/46551/p2p/12D3KooWMuNE7uQczAengmmxXzSRjgoRuer1PqJRHTeY6NrgF5R9"
EXPERT_UID = "vgg16_large.2"  # Second half of VGG16
NUM_CLASSES = 10  # CIFAR-10 has 10 classes

class VGG16SecondHalf(nn.Module):
    """Second half of VGG16: features[23:] + classifier (Blocks 4-5 + FC layers)"""
    def __init__(self, num_classes=10):
        super().__init__()
        # Load the full VGG16 model
        vgg16 = models.vgg16(pretrained=False)
        
        # Take the second half of features (blocks 4-5)
        self.features = nn.Sequential(*list(vgg16.features.children())[23:])
        
        # Take the classifier (FC layers)
        self.avgpool = vgg16.avgpool
        self.classifier = vgg16.classifier
        
        # Modify the last layer for CIFAR-10
        self.classifier[6] = nn.Linear(4096, num_classes)
        
        # Print the architecture
        print("VGG16 Second Half Architecture:")
        print("Features:")
        print(self.features)
        print("Classifier:")
        print(self.classifier)
        
        # Calculate parameters
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {num_params:,}")
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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

# Initialize the second half of VGG16 with CIFAR-10's 10 classes
model = VGG16SecondHalf(num_classes=NUM_CLASSES).to(device)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Define input/output shapes
# Input: [batch_size, 256, 28, 28] - Output from first half
# Output: [batch_size, 10] - Classification output (10 classes)
input_shape = (None, 256, 28, 28)
output_shape = (None, NUM_CLASSES)

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