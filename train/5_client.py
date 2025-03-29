import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import hivemind
import time
from tqdm import tqdm

# Configuration for full dataset run
DHT_ADDR = "/ip4/34.60.208.208/tcp/46551/p2p/12D3KooWMuNE7uQczAengmmxXzSRjgoRuer1PqJRHTeY6NrgF5R9"
EXPERT1_UID = "vgg16_large.1"  # First half of VGG16
EXPERT2_UID = "vgg16_large.2"  # Second half of VGG16
NUM_CLASSES = 10
BATCH_SIZE = 32
NUM_EPOCHS = 1

# Create a class that combines the two remote experts
class DistributedVGG16(nn.Module):
    def __init__(self, first_half, second_half):
        super().__init__()
        self.first_half = first_half
        self.second_half = second_half
    
    def forward(self, x):
        x = self.first_half(x)
        x = self.second_half(x)
        return x

# Test DHT connection before proceeding
print(f"Testing connection to DHT at {DHT_ADDR}...")
try:
    dht = hivemind.DHT(
        host_maddrs=["/ip4/0.0.0.0/tcp/0"],
        initial_peers=[DHT_ADDR],
        start=True,
        client_mode=True
    )
    print("✓ Successfully connected to DHT!")
    
    # Try to print DHT peers if possible
    try:
        peers = dht.get_alive_peers()
        print(f"Found {len(peers)} peers in the DHT")
    except:
        print("Connected to DHT but could not query peers")
    
    print("My peer addresses:")
    for addr in dht.get_visible_maddrs():
        print(str(addr))
except Exception as e:
    print(f"✗ Failed to connect to DHT: {str(e)}")
    raise

# Find the first expert
print(f"Looking for expert: {EXPERT1_UID}")
first_half = None
for attempt in range(10):
    try:
        experts = hivemind.moe.get_experts(dht, [EXPERT1_UID])
        first_half = experts[0]
        if first_half is not None:
            print(f"✓ Successfully connected to expert: {EXPERT1_UID}")
            break
        else:
            print(f"✗ Expert {EXPERT1_UID} not found (attempt {attempt+1})")
    except Exception as e:
        print(f"× Attempt {attempt+1} failed: {str(e)}")
    print("Retrying in 5 seconds...")
    time.sleep(5)

if first_half is None:
    raise RuntimeError(f"Could not find expert: {EXPERT1_UID}")

# Find the second expert
print(f"Looking for expert: {EXPERT2_UID}")
second_half = None
for attempt in range(10):
    try:
        experts = hivemind.moe.get_experts(dht, [EXPERT2_UID])
        second_half = experts[0]
        if second_half is not None:
            print(f"✓ Successfully connected to expert: {EXPERT2_UID}")
            break
        else:
            print(f"✗ Expert {EXPERT2_UID} not found (attempt {attempt+1})")
    except Exception as e:
        print(f"× Attempt {attempt+1} failed: {str(e)}")
    print("Retrying in 5 seconds...")
    time.sleep(5)

if second_half is None:
    raise RuntimeError(f"Could not find expert: {EXPERT2_UID}")

# Try a test forward pass to verify connection
print("Testing connection with a single forward pass...")
try:
    dummy_input = torch.randn(1, 3, 224, 224)
    test_output = first_half(dummy_input)
    print(f"First half output shape: {test_output.shape}")
    final_output = second_half(test_output)
    print(f"Second half output shape: {final_output.shape}")
    print("✓ Test forward pass successful!")
except Exception as e:
    print(f"✗ Test forward pass failed: {str(e)}")
    print("Continuing anyway, but expect potential issues during training")

# Create the distributed model
model = DistributedVGG16(first_half, second_half)
print("Successfully created distributed VGG16 model")

# Prepare the full CIFAR-10 dataset
print("Loading dataset...")

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize(224),  # VGG16 expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the full CIFAR-10 dataset
dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

# Create data loader with the full dataset
train_loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=2  # Parallel data loading
)

print(f"Dataset prepared with {len(dataset)} images")

# Set up loss function
criterion = nn.CrossEntropyLoss()

# Training loop
print(f"Starting training for {NUM_EPOCHS} epochs...")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Track batch times to estimate completion
    batch_times = []
    batch_sizes = []
    successful_batches = 0
    failed_batches = 0
    
    # Initialize these to default values in case all batches fail
    avg_loss = 0.0
    accuracy = 0.0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for i, (inputs, labels) in enumerate(progress_bar):
        batch_start = time.time()
        
        try:
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Backward pass (will update the remote experts)
            loss.backward()
            
            # Update statistics
            running_loss += loss.item()
            avg_loss = running_loss / (successful_batches + 1)
            accuracy = 100 * correct / total
            
            # Record successful batch
            successful_batches += 1
            
            # Calculate batch time
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            batch_sizes.append(labels.size(0))
            
            # Calculate ETA
            if len(batch_times) > 1:
                avg_batch_time = sum(batch_times) / len(batch_times)
                batches_left = len(train_loader) - (i + 1)
                eta_seconds = avg_batch_time * batches_left
                eta_min = eta_seconds // 60
                eta_sec = eta_seconds % 60
                eta_str = f"{int(eta_min)}m {int(eta_sec)}s"
            else:
                eta_str = "Calculating..."
            
            # Update progress bar
            progress_bar.set_postfix({
                "Loss": f"{avg_loss:.4f}",
                "Acc": f"{accuracy:.2f}%",
                "Batch": f"{batch_time:.2f}s",
                "ETA": eta_str
            })
            
        except Exception as e:
            failed_batches += 1
            print(f"\nError during batch {i}: {str(e)}")
            print("Continuing to next batch...")
    
    # Print epoch statistics
    epoch_time = time.time() - start_time
    
    if successful_batches > 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    else:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - No successful batches!")
    
    print(f"Epoch completed in {epoch_time//60:.0f}m {epoch_time%60:.0f}s")
    print(f"Successful batches: {successful_batches}, Failed batches: {failed_batches}")
    
    if batch_times:
        avg_batch_time = sum(batch_times) / len(batch_times)
        print(f"Average batch time: {avg_batch_time:.2f}s")
        print(f"Processing speed: {sum(batch_sizes)/sum(batch_times):.2f} images/second")
    else:
        print("No successful batches to compute statistics")

# Calculate total training time
training_time = time.time() - start_time
print(f"Training completed in {training_time//60:.0f}m {training_time%60:.0f}s")

# Test with a random batch
def test_model(model, num_samples=16):
    # Get some test images
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=num_samples, 
        shuffle=True
    )
    
    # Get class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Make predictions
    try:
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
        
        # Calculate accuracy on this batch
        correct = (predicted == labels).sum().item()
        accuracy = 100 * correct / labels.size(0)
        print(f"Test batch accuracy: {accuracy:.2f}% ({correct}/{labels.size(0)})")
        
        # Print results for the first 5 images
        print("\nSample Predictions:")
        for i in range(min(5, num_samples)):
            print(f"True: {classes[labels[i]]}, Predicted: {classes[predicted[i]]}")
    except Exception as e:
        print(f"Error during testing: {str(e)}")

# Test the model with a few samples
try:
    print("\nTesting model with test samples...")
    test_model(model)
except Exception as e:
    print(f"Error during testing: {str(e)}")

# Clean up
print("Shutting down client...")
dht.shutdown()
print("Client shutdown complete")