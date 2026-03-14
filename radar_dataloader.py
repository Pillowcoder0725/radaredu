import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def create_radar_dataloaders(data_dir, batch_size=16, train_split=0.8):
    """
    Creates PyTorch DataLoaders for the radar spectrogram dataset.
    Optimized for grayscale images to feed into the 1-channel hierarchical Vision Transformer.
    """
    print(f"Scanning dataset directory: {data_dir}...")
    
    # Define transformations
    # 1. Grayscale: Forces 1-channel output (crucial for our modified Swin model)
    # 2. Resize: Ensures strict 256x256 dimensions
    # 3. ToTensor: Converts pixel values (0-255) to float tensors (0.0-1.0)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    try:
        # Load the dataset
        # ImageFolder automatically infers classes from the subdirectories
        full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        print(f"Success! Found {len(full_dataset)} images.")
        print(f"Detected {len(full_dataset.classes)} classes: {full_dataset.classes}")
    except FileNotFoundError:
        print(f"Error: Could not find directory '{data_dir}'. Please run generate_dataset.py first.")
        return None, None, None
        
    # Split into training and validation sets for unbiased evaluation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Use a fixed generator seed for reproducible splits across different runs
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Create the DataLoaders
    # Keeping batch sizes relatively small is often necessary when simulating or targeting 
    # resource-constrained edge devices with limited memory.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, full_dataset.classes

if __name__ == "__main__":
    # Test the dataloader with the directory created in the previous step
    dataset_path = "radar_dataset"
    
    # Initializing with a small batch size
    train_loader, val_loader, classes = create_radar_dataloaders(dataset_path, batch_size=4)
    
    if train_loader:
        print("\n--- Testing DataLoader Output ---")
        # Fetch one single batch of data from the pipeline
        images, labels = next(iter(train_loader))
        
        print(f"Batch image tensor shape: {images.shape}")
        print(f"Expected shape: (Batch_Size, 1, 256, 256)")
        print(f"\nBatch label tensor shape: {labels.shape}")
        print(f"Sample labels (indices): {labels.tolist()}")
        print(f"Sample labels (names): {[classes[label] for label in labels]}")
        
        print("\nIf the shapes match, your data is perfectly formatted and ready for inference.")
