import torch
import torch.nn as nn
import torch.optim as optim
from pyradar_edu import RadarProcessor
from student_swin import RadarSwinTransformer

class RadarPipeline:
    def __init__(self, num_classes=5, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RadarSwinTransformer(num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize radar DSP module
        self.radar_dsp = RadarProcessor(sample_rate=2e6, sweep_time=40e-6, num_samples=256)

    def train_step(self, spectrogram_batch, labels):
        """Performs one training iteration."""
        self.model.train()
        spectrogram_batch = spectrogram_batch.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(spectrogram_batch)
        
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def process_and_predict(self, raw_radar_matrix):
        """End-to-end pipeline: DSP to Machine Learning inference."""
        self.model.eval()
        
        # 1. Apply FFTs to get micro-Doppler spectrogram
        spectrogram = self.radar_dsp.apply_doppler_fft(raw_radar_matrix)
        
        # 2. Convert to PyTorch tensor and add batch/channel dimensions
        tensor_spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        tensor_spectrogram = tensor_spectrogram.unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 3. Model Inference
        with torch.no_grad():
            predictions = self.model(tensor_spectrogram)
            predicted_class = torch.argmax(predictions, dim=1).item()
            
        return predicted_class
