import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data_dir = "dataset/split"                                 # Path to dataset folder (split into train/val/test)
model_path = "EfficientNetB3/train5/best_model_loss.pt"    # Path to trained model weights
num_classes = 7                                            # Number of target classes
batch_size = 32                                             
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

exclude_uncertain = True                                   # Whether to exclude low-confidence predictions
uncertainty_threshold = 0.80                               # Minimum confidence to accept a prediction

# Image Preprocessing (same as during training) 
transform = transforms.Compose([
    transforms.Resize((300, 300)),                          
    transforms.ToTensor(),                              
    transforms.Normalize([0.485, 0.456, 0.406],             
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names = test_dataset.classes                         # Get class labels from folder structure

class EfficientNetB3SkinLesion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.features = base_model.features                 
        self.pooling = base_model.avgpool                  
        self.classifier = nn.Sequential(                     
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(base_model.classifier[1].in_features, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# === Load model ===
model = EfficientNetB3SkinLesion(num_classes).to(device)
model.load_state_dict(torch.load(model_path))               # Load trained weights
model.eval()                                                # Set model to eval mode (no dropout, etc.)


all_preds = []              # Predicted class indices
all_labels = []             # True labels
all_pred_names = []         # Class name strings
all_confidences = []        # Confidence values
uncertain_count = 0         # Number of uncertain predictions

with torch.no_grad():                                   
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)             

        for i in range(probs.size(0)):
            label = labels[i].item()                        
            prob_row = probs[i]                             # Probability distribution over classes
            top_confidence = prob_row.max().item()          # Highest class probability
            top_class = prob_row.argmax().item()            # Predicted class

            all_labels.append(label)

            if top_confidence >= uncertainty_threshold:
                # if confident prediction then accept
                all_preds.append(top_class)
                all_pred_names.append(class_names[top_class])
                all_confidences.append(top_confidence)
            elif not exclude_uncertain:
                # Accept even if uncertain
                all_preds.append(top_class)
                all_pred_names.append(class_names[top_class] + " (uncertain)")
                all_confidences.append(top_confidence)
                uncertain_count += 1
            else:
                # Skip prediction (uncertain)
                all_preds.append(None)
                all_pred_names.append("Uncertain")
                all_confidences.append(top_confidence)
                uncertain_count += 1

# Filters out uncertain predictions if needed
filtered_preds = [p for p in all_preds if p is not None]
filtered_labels = [l for p, l in zip(all_preds, all_labels) if p is not None]

# Calculates accuracy
correct = sum(p == l for p, l in zip(filtered_preds, filtered_labels))
total = len(filtered_preds)
acc = correct / total if total > 0 else 0

print(f"\nAccuracy ({'excluding' if exclude_uncertain else 'including'} uncertain): {acc:.4f}")
print(f"Uncertain predictions: {uncertain_count} out of {len(all_labels)}")

# Prints classification report
print("\n=== Classification Report ===")
print(classification_report(filtered_labels, filtered_preds, target_names=class_names))

# Plot and save confusion matrix 
cm = confusion_matrix(filtered_labels, filtered_preds, labels=list(range(num_classes)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix" + (" (Filtered)" if exclude_uncertain else " (All Predictions)"))
plt.tight_layout()
plt.savefig("EfficientNetB3/confusion_matrix_80_loss.png")
plt.show()

# Save predictions to CSV
df = pd.DataFrame({
    "True Label": [class_names[i] for i in all_labels],     # Original label from dataset
    "Predicted Label": all_pred_names,                      # Predicted class name or "Uncertain"
    "Confidence": all_confidences                           # Max softmax probability
})
df.to_csv("EfficientNetB3/predictions.csv", index=False)
print("Saved predictions to EfficientNetB3/predictions.csv")