import torch
import numpy as np
from scipy import signal
import torch.nn.functional as F
import time
import os
from student_swin import RadarSwinTransformer

class RadarInference:
    def __init__(self, model_path, class_names, sample_rate=2e6):
        # Automatically use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        self.class_names = class_names
        
        print(f"Loading model weights from '{model_path}' onto {self.device}...")
        
        # Initialize the model architecture (must match the trained model exactly)
        self.model = RadarSwinTransformer(num_classes=len(class_names), in_channels=1)
        
        # Load the saved state dictionary
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        except FileNotFoundError:
            print(f"Error: Could not find '{model_path}'. Did you run the training script first?")
            exit()
            
        self.model.to(self.device)
        self.model.eval() # CRITICAL: Disables dropout and batch norm updates for inference
        print("Success! Model loaded and ready for real-time inference.\n")

    def preprocess_signal(self, raw_adc_data):
        """Processes a raw 1D radar array directly into a 256x256 PyTorch tensor."""
        # 1. Compute STFT (Matching your training parameters exactly)
        frequencies, times, Zxx = signal.stft(
            raw_adc_data, fs=self.sample_rate, nperseg=2048, noverlap=1024, return_onesided=False
        )
        
        Zxx = np.fft.fftshift(Zxx, axes=0)
        spectrogram_db = 10 * np.log10(np.abs(Zxx) + 1e-10)
        
        # 2. Normalize the matrix to a 0.0 - 1.0 range (mimicking torchvision.transforms.ToTensor)
        spec_min, spec_max = spectrogram_db.min(), spectrogram_db.max()
        if spec_max > spec_min:
            normalized_spec = (spectrogram_db - spec_min) / (spec_max - spec_min)
        else:
            normalized_spec = spectrogram_db - spec_min
        
        # 3. Convert to tensor and add batch/channel dimensions -> Shape: (1, 1, H, W)
        tensor_spec = torch.tensor(normalized_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # 4. Dynamically interpolate to 256x256 pixels to match the Swin Transformer input
        resized_tensor = F.interpolate(tensor_spec, size=(256, 256), mode='bilinear', align_corners=False)
        
        return resized_tensor

    def predict(self, raw_adc_data):
        """Runs the end-to-end inference pipeline and calculates confidence scores."""
        start_time = time.time()
        
        # Preprocess and move to device
        input_tensor = self.preprocess_signal(raw_adc_data).to(self.device)
        
        # Run forward pass without tracking gradients (saves memory and speeds up execution)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Convert raw logits into probabilities that sum to 100%
            probabilities = torch.softmax(outputs, dim=1)[0]
            
            # Extract the index of the highest probability
            predicted_idx = torch.argmax(probabilities).item()
            
        inference_time = (time.time() - start_time) * 1000 # Convert to milliseconds
        
        predicted_class = self.class_names[predicted_idx]
        confidence = probabilities[predicted_idx].item() * 100
        
        return predicted_class, confidence, inference_time

if __name__ == "__main__":
    # --- Configuration ---
    # Update this list to exactly match the folder names inside your 'radar_dataset' 
    # Must be in the same alphabetical order that PyTorch ImageFolder used!
    MY_CLASSES = ["baseline", "cognitive_load", "drowsy"] 
    MODEL_WEIGHTS = "best_radar_swin.pth"
    
    # Initialize the inference engine
    infer_engine = RadarInference(model_path=MODEL_WEIGHTS, class_names=MY_CLASSES)
    
    # --- Simulating a live radar feed ---
    print("Capturing new raw radar signal...")
    
    # Generating 5 seconds of a dummy physiological signal for testing
    t = np.linspace(0, 5, int(2e6 * 5))
    dummy_live_signal = np.cos(2 * np.pi * 500 * t) * np.cos(2 * np.pi * 1.2 * t) + np.random.normal(0, 0.2, len(t))
    
    # --- Run Inference ---
    prediction, confidence, latency = infer_engine.predict(dummy_live_signal)
    
    print("-" * 40)
    print("INFERENCE RESULTS:")
    print(f"Detected State : {prediction.upper()}")
    print(f"Confidence     : {confidence:.2f}%")
    print(f"Latency        : {latency:.2f} ms")
    print("-" * 40)
