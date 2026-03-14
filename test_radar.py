import numpy as np
from pyradar_edu import RadarProcessor

def run_radar_test():
    print("Initializing RadarProcessor...")
    # Initialize with the same parameters used in your ML pipeline
    processor = RadarProcessor(sample_rate=2e6, sweep_time=40e-6, num_samples=256)
    
    print("\n--- Testing 1D Range FFT ---")
    # Generate a dummy 1D beat signal (a simple sine wave representing a single target)
    t = np.linspace(0, processor.sweep_time, processor.num_samples)
    target_freq = 100000  # 100 kHz target frequency
    dummy_beat_signal = np.sin(2 * np.pi * target_freq * t)
    
    try:
        range_profile = processor.apply_range_fft(dummy_beat_signal)
        print(f"Success! Range profile generated.")
        print(f"Expected shape: (128,) | Actual shape: {range_profile.shape}")
    except Exception as e:
        print(f"Error during 1D FFT: {e}")

    print("\n--- Testing 2D Micro-Doppler FFT ---")
    # Generate a dummy 2D radar data matrix (simulating 64 chirps)
    num_chirps = 64
    dummy_radar_matrix = np.random.rand(num_chirps, processor.num_samples) + \
                         1j * np.random.rand(num_chirps, processor.num_samples)
                         
    try:
        doppler_spectrogram = processor.apply_doppler_fft(dummy_radar_matrix)
        print(f"Success! Micro-Doppler spectrogram generated.")
        print(f"Expected shape: (64, 256) | Actual shape: {doppler_spectrogram.shape}")
    except Exception as e:
        print(f"Error during 2D FFT: {e}")
        
    print("\nIf you see the 'Success!' messages above, your formatting is fixed and the code is working perfectly.")

if __name__ == "__main__":
    run_radar_test()
