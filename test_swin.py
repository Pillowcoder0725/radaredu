import torch
from student_swin import RadarSwinTransformer

def run_swin_test():
    print("Initializing RadarSwinTransformer...")
    
    try:
        # Initialize with 5 classes and 1 input channel (for grayscale spectrograms)
        model = RadarSwinTransformer(num_classes=5, in_channels=1)
        print("Success! Model initialized and layers modified.")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    print("\n--- Testing Forward Pass ---")
    
    # Create a dummy tensor representing a single grayscale spectrogram
    # Shape: (Batch Size, Channels, Height, Width)
    # Swin_V2_T typically expects 256x256 inputs
    batch_size = 1
    channels = 1
    height = 256
    width = 256
    
    dummy_spectrogram = torch.randn(batch_size, channels, height, width)
    print(f"Feeding dummy spectrogram of shape: {dummy_spectrogram.shape}")

    try:
        # Set model to evaluation mode (disables dropout, etc.)
        model.eval()
        
        # Turn off gradient calculation for a simple inference test
        with torch.no_grad():
            output = model(dummy_spectrogram)
            
        print(f"Success! Forward pass completed.")
        print(f"Expected output shape: ({batch_size}, 5) | Actual shape: {list(output.shape)}")
        
        # Print the raw logits (unnormalized predictions)
        print(f"Raw output logits: {output.numpy()}")
        
    except Exception as e:
        print(f"Error during forward pass: {e}")

    print("\nIf you see the 'Success!' messages above, PyTorch is correctly configured and the model is ready.")

if __name__ == "__main__":
    run_swin_test()
