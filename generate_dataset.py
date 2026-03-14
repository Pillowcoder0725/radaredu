import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

def create_spectrogram_image(raw_adc_data, save_path, sample_rate=2e6):
    """
    Processes raw radar signals via STFT and saves a clean, 
    axis-free PNG image optimized for Vision Transformers.
    """
    # 1. Calculate Short-Time Fourier Transform (STFT)
    frequencies, times, Zxx = signal.stft(
        raw_adc_data, 
        fs=sample_rate, 
        nperseg=2048, 
        noverlap=1024,
        return_onesided=False
    )
    
    # Shift frequencies and convert to dB magnitude
    frequencies = np.fft.fftshift(frequencies)
    Zxx = np.fft.fftshift(Zxx, axes=0)
    spectrogram_db = 10 * np.log10(np.abs(Zxx) + 1e-10)

    # 2. Configure Matplotlib for pure image generation
    # The figsize matches standard Swin Transformer input ratios
    fig = plt.figure(figsize=(2.56, 2.56), dpi=100) 
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off() # Crucial: Turns off axes and labels
    fig.add_axes(ax)

    # 3. Plot and save
    # We constrain the y-axis to the relevant micro-Doppler ranges 
    ax.pcolormesh(times, frequencies, spectrogram_db, shading='gouraud', cmap='jet')
    ax.set_ylim(-1000, 1000) 

    # Save directly to the provided path
    plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig) # Free up memory 

def process_directory(input_folder, output_base_folder, class_name):
    """
    Loops through all raw data files in a folder and converts them to PNGs.
    """
    # Ensure the output directory exists
    output_dir = os.path.join(output_base_folder, class_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing raw files for class: '{class_name}'...")
    
    # Optional: Replace this with your actual file loading logic
    # For demonstration, we will simulate finding two raw data files
    dummy_files = [f"sample_01_{class_name}.npy", f"sample_02_{class_name}.npy"]
    
    for i, file_name in enumerate(dummy_files):
        # In a real scenario, you would load your actual data here:
        # raw_data = np.load(os.path.join(input_folder, file_name))
        
        # Simulating raw radar data for the sake of the script running
        t = np.linspace(0, 5, int(2e6 * 5))
        raw_data = np.cos(2 * np.pi * 500 * t) * np.cos(2 * np.pi * 1.5 * t) + np.random.normal(0, 0.2, len(t))
        
        # Generate the save path
        save_filename = f"{class_name}_{i:04d}.png"
        save_path = os.path.join(output_dir, save_filename)
        
        # Create and save the image
        create_spectrogram_image(raw_data, save_path)
        print(f" -> Saved {save_filename}")

if __name__ == "__main__":
    # Define your dataset structure
    base_dataset_dir = "radar_dataset"
    
    # Process distinct classes (e.g., baseline vs. cognitive load vs. drowsiness)
    # You would point the first argument to the folder containing your raw .npy or .csv files
    process_directory("raw_data/baseline", base_dataset_dir, "baseline")
    process_directory("raw_data/drowsy", base_dataset_dir, "drowsy")
    
    print(f"\nSuccess! Dataset generated at: ./{base_dataset_dir}/")
