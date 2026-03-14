
# Radar-Ed: FMCW Radar Machine Learning Pipeline

This repository contains an end-to-end Python pipeline for processing raw Frequency-Modulated Continuous-Wave (FMCW) radar data into micro-Doppler spectrograms and classifying them using a customized, single-channel Swin Transformer.

The architecture bridges digital signal processing (DSP) and deep learning, optimized for resource-constrained environments.

## Repository Structure

The project is divided into core architecture, data preparation, training, and deployment scripts.

### Core Architecture
`pyradar_edu.py`: The core digital signal processing (DSP) module. Contains the critical mathematics for windowing and applying 1D Range and 2D Micro-Doppler Fast Fourier Transforms (FFTs) to extract kinematic signatures.
`student_swin.py`: A PyTorch hierarchical vision transformer adapted to accept 1-channel (grayscale) spectrograms instead of standard 3-channel RGB images.
`radar_ml_pipeline.py`: The operational bridge combining the DSP module and the Swin Transformer. It handles the full training loop, tracking loss and accuracy metrics across epochs.
### Data Preparation & Visualization
`visualize_radar.py`: Processes raw analog-to-digital (ADC) signals using a Short-Time Fourier Transform (STFT) to generate high-resolution, time-velocity spectrograms via Matplotlib.
`generate_dataset.py`: A batch-processing script that converts raw radar signals into clean, axis-free PNG images and saves them directly into a PyTorch-ready directory structure.
`radar_dataloader.py`: Creates PyTorch DataLoaders optimized for streaming grayscale spectrogram images directly into the 1-channel vision transformer.
### Deployment
`run_inference.py`: A lightweight, real-time inference script designed for production. It computes the STFT matrix purely mathematically in memory, dynamically resizes it into a PyTorch tensor, and bypasses heavy image-saving steps to minimize latency.

### Testing Utilities

These scripts verify the structural integrity of the environment and ensure the mathematical functions execute without errors.
`test_radar.py`**: Verifies the 1D and 2D FFT logic.
`test_swin.py`**: Ensures the Swin Transformer's modified first layer accepts single-channel inputs and executes a forward pass.
`test_pipeline.py`**: Confirms the end-to-end bridge between the DSP script and the PyTorch model.

## 🚀 Workflow & Usage

Follow these steps to process data, train the model, and deploy it for inference.

### 1. Verify Environment

Run the test scripts to ensure PyTorch and NumPy are configured correctly:
python test_radar.py
python test_swin.py
python test_pipeline.py

### 2. Visualize Your Data

Check your physical sensor data by generating a micro-Doppler time-velocity spectrogram. If no data is provided, the script will simulate a physiological signal:

python visualize_radar.py

### 3. Generate the Training Dataset

Batch-process your raw `.npy` or `.csv` files into PyTorch-ready image folders. Ensure you define your specific classes inside the script before running:

python generate_dataset.py

### 4. Train the Model

Train the Swin Transformer using the generated dataset. The pipeline will automatically stream the data, track validation accuracy, and save the optimal weights to `best_radar_swin.pth`:

python radar_ml_pipeline.py

### 5. Run Real-Time Inference

Load your optimized `best_radar_swin.pth` weights and run classification on a live radar feed or a newly captured signal:

python run_inference.py
