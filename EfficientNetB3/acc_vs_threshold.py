import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data_dir = "dataset/split"                         # Directory where your test data is stored
model_path = "EfficientNetB3/best_model_acc.pt"    # Path to your saved model
num_classes = 7                                    # Number of output classes
batch_size = 32                                     
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
exclude_uncertain = True                            # If True, exclude uncertain predictions from accuracy


transform = transforms.Compose([
    transforms.Resize((300, 300)),                  # Resize to match EfficientNet input size
    transforms.ToTensor(),                        
    transforms.Normalize([0.485, 0.456, 0.406],     
                         [0.229, 0.224, 0.225])
])

# Loads test dataset
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names = test_dataset.classes                  

# Define the model architecture
class EfficientNetB3SkinLesion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.features = base_model.features         # Feature extraction layers
        self.pooling = base_model.avgpool           # Global average pooling
        self.classifier = nn.Sequential(           
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(base_model.classifier[1].in_features, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)                     
        return self.classifier(x)

# === Load the model weights ===
model = EfficientNetB3SkinLesion(num_classes).to(device)
model.load_state_dict(torch.load(model_path))        
model.eval()                                         # Sets the model to evaluation mode

# === Evaluation loop over different confidence thresholds ===
thresholds = np.arange(0.50, 0.91, 0.05)            # Range of thresholds: 0.50 to 0.90 in steps of 0.05
accuracies = []                                     # To store accuracy at each threshold
uncertains = []                                     # To store count of uncertain predictions

with torch.no_grad():                              
    for threshold in thresholds:
        all_preds = []                              
        all_labels = []                            
        uncertain_count = 0                         # Counter for uncertain predictions

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)                 # Forward pass
            probs = torch.softmax(outputs, dim=1)   

            for i in range(probs.size(0)):
                label = labels[i].item()            
                prob_row = probs[i]                 
                top_confidence = prob_row.max().item()  # Highest probability
                top_class = prob_row.argmax().item()    # Class index with highest probability

                all_labels.append(label)

                if top_confidence >= threshold:     # Only accept prediction if confidence is high enough
                    all_preds.append(top_class)
                elif not exclude_uncertain:         # If not excluding uncertain, accept anyway
                    all_preds.append(top_class)
                else:                               # Otherwise mark as uncertain
                    all_preds.append(None)
                    uncertain_count += 1

        # Filters out uncertain predictions
        filtered_preds = [p for p in all_preds if p is not None]
        filtered_labels = [l for p, l in zip(all_preds, all_labels) if p is not None]

        # Computes the accuracy 
        correct = sum(p == l for p, l in zip(filtered_preds, filtered_labels))
        total = len(filtered_preds)
        acc = correct / total if total > 0 else 0

        accuracies.append(acc)
        uncertains.append(uncertain_count)

# Plot accuracy vs. threshold
plt.figure(figsize=(8, 5))
plt.plot(thresholds, accuracies, marker='o', label="Accuracy (Excluding Uncertain)")
plt.xlabel("Uncertainty Threshold")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Uncertainty Threshold")
plt.xticks(thresholds)
plt.grid(True)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("EfficientNetB3/accuracy_vs_threshold_acc.png")  # Save plot
plt.show()

print("\n=== Threshold Summary ===")
for t, acc, u in zip(thresholds, accuracies, uncertains):
    print(f"Threshold {t:.2f} → Accuracy: {acc:.4f}, Uncertain: {u} ({u/len(test_dataset):.1%})")