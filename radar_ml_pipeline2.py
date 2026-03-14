import torch
import torch.nn as nn
import torch.optim as optim
import time
from pyradar_edu import RadarProcessor
from student_swin import RadarSwinTransformer
from radar_dataloader import create_radar_dataloaders

class RadarPipeline:
    def __init__(self, num_classes=5, learning_rate=0.001):
        # Automatically use GPU acceleration if available, otherwise fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RadarSwinTransformer(num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize radar DSP module
        self.radar_dsp = RadarProcessor(sample_rate=2e6, sweep_time=40e-6, num_samples=256)

    def train_model(self, data_dir, epochs=10, batch_size=16):
        """Full training loop tracking loss and accuracy across epochs."""
        print(f"Initializing training on device: {self.device}")
        
        # Load the automated streaming pipelines we built
        train_loader, val_loader, classes = create_radar_dataloaders(data_dir, batch_size=batch_size)
        if not train_loader:
            return

        # Ensure the model's output layer matches the number of detected classes
        if len(classes) != self.model.swin.head.out_features:
            print(f"Adjusting output layer from {self.model.swin.head.out_features} to {len(classes)} classes...")
            in_features = self.model.swin.head.in_features
            self.model.swin.head = nn.Linear(in_features, len(classes)).to(self.device)

        best_val_accuracy = 0.0

        for epoch in range(epochs):
            start_time = time.time()
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            # --- Training Phase ---
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Track metrics
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_loss = running_loss / total_train
            train_acc = correct_train / total_train

            # --- Validation Phase ---
            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad(): # Disable gradient tracking for faster evaluation
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_loss = val_loss / total_val
            val_acc = correct_val / total_val
            epoch_time = time.time() - start_time

            # Print epoch summary
            print(f"\nEpoch [{epoch+1}/{epochs}] | Time: {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.1f}%")

            # Save the optimal weights
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(self.model.state_dict(), "best_radar_swin.pth")
                print("  -> Metric improved! Model saved to 'best_radar_swin.pth'")

        print(f"\nTraining sequence complete! Best Validation Accuracy achieved: {best_val_accuracy*100:.1f}%")

if __name__ == "__main__":
    # To run this, simply execute `python radar_ml_pipeline.py`
    # Make sure you have your 'radar_dataset' folder generated from the previous steps
    pipeline = RadarPipeline()
    pipeline.train_model(data_dir="radar_dataset", epochs=5, batch_size=4)
