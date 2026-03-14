import numpy as np
import torch
from radar_ml_pipeline import RadarPipeline

def run_pipeline_test():
    print("Initializing Full Radar ML Pipeline...")
    try:
        # Initialize the pipeline (which automatically sets up the DSP processor and Swin model)
        pipeline = RadarPipeline(num_classes=5)
        print("Success! Pipeline initialized with DSP and Swin Transformer components.")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return

    print("\n--- Testing End-to-End Inference ---")
    # Generate dummy raw radar data simulating FMCW chirps
    # Shape: (Number of Chirps, Samples per Chirp)
    num_chirps = 64
    num_samples = 256
    dummy_raw_radar = np.random.rand(num_chirps, num_samples) + \
                      1j * np.random.rand(num_chirps, num_samples)
    
    print(f"Feeding raw complex radar matrix of shape: {dummy_raw_radar.shape}")

    try:
        # Run the full sequence: raw data -> DSP spectrogram -> Swin Model -> prediction
        prediction = pipeline.process_and_predict(dummy_raw_radar)
        print(f"Success! End-to-end inference completed.")
        print(f"Predicted Class Index: {prediction}")
    except Exception as e:
        print(f"Error during inference: {e}")

    print("\n--- Testing Single Training Step ---")
    # Generate a dummy batch of pre-processed spectrograms to test the training loop
    # Shape: (Batch Size, Channels, Height, Width)
    batch_size = 4
    dummy_spectrogram_batch = torch.randn(batch_size, 1, 256, 256)
    
    # Generate random dummy labels for the batch (values between 0 and 4)
    dummy_labels = torch.randint(0, 5, (batch_size,))

    try:
        loss = pipeline.train_step(dummy_spectrogram_batch, dummy_labels)
        print("Success! Training step completed.")
        print(f"Calculated Loss: {loss:.4f}")
    except Exception as e:
        print(f"Error during training step: {e}")

    print("\nIf all tests say 'Success!', your entire radar machine learning pipeline is fully operational.")

if __name__ == "__main__":
    run_pipeline_test()
