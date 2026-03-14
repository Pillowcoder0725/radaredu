import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

def load_and_visualize_stft(file_path, sample_rate=2e6):
    """
    Loads raw analog-to-digital radar signals and generates a 
    time-velocity (micro-Doppler) spectrogram using STFT.
    """
    print(f"Attempting to load raw radar data from: {file_path}...")
    
    # 1. Load the Data
    if os.path.exists(file_path):
        # Assumes data is a 1D numpy array of raw ADC samples. 
        # Update np.load to np.loadtxt or pd.read_csv if using a different format.
        raw_adc_data = np.load(file_path)
        print("Success! Real dataset loaded.")
    else:
        print(f"File not found. Generating a simulated physiological signal to demonstrate the STFT plot...")
        # Simulating a 5-second signal with a base frequency and low-frequency micro-Doppler modulation 
        t = np.linspace(0, 5, int(sample_rate * 5))
        carrier = np.cos(2 * np.pi * 500 * t) 
        modulation = np.cos(2 * np.pi * 1.5 * t) # Simulating a 1.5 Hz motion 
        raw_adc_data = carrier * modulation + np.random.normal(0, 0.2, len(t))

    print("Processing raw signals into a time-velocity spectrogram via STFT...")
    
    # 2. Calculate Short-Time Fourier Transform (STFT)
    # nperseg controls the window size. Adjust this to trade off between time and frequency resolution.
    frequencies, times, Zxx = signal.stft(
        raw_adc_data, 
        fs=sample_rate, 
        nperseg=2048, 
        noverlap=1024,
        return_onesided=False # Keep negative frequencies to show directional velocity (towards/away)
    )
    
    # Shift the zero frequency (stationary objects) to the vertical center of the plot
    frequencies = np.fft.fftshift(frequencies)
    Zxx = np.fft.fftshift(Zxx, axes=0)
    
    # Convert the complex STFT output to a logarithmic magnitude scale (dB) for visual clarity
    magnitude_spectrogram = np.abs(Zxx)
    spectrogram_db = 10 * np.log10(magnitude_spectrogram + 1e-10)

    print("Rendering visualization...")
    
    # 3. Plot the Spectrogram
    plt.figure(figsize=(12, 6))
    
    # Use pcolormesh for a continuous heat map look
    plt.pcolormesh(times, frequencies, spectrogram_db, shading='gouraud', cmap='jet')
    
    plt.title('Micro-Doppler Time-Velocity Spectrogram')
    plt.ylabel('Doppler Frequency Shift / Velocity (Hz)')
    plt.xlabel('Time (Seconds)')
    
    # Zoom in on the low-frequency micro-Doppler range where human motion occurs
    # You may need to tweak these limits depending on your specific target speeds
    plt.ylim(-1000, 1000) 
    
    plt.colorbar(label='Signal Magnitude (dB)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Point this to your actual raw data file
    target_file = 'my_raw_radar_data.npy'
    load_and_visualize_stft(target_file)
