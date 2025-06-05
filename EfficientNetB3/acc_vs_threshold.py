import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === CONFIG ===
data_dir = "dataset/split"
model_path = "EfficientNetB3/best_model_acc.pt"
num_classes = 7
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exclude_uncertain = True  # Only include confident predictions

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === DATA ===
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names = test_dataset.classes

# === MODEL ===
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

model = EfficientNetB3SkinLesion(num_classes).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# === THRESHOLD SWEEP ===
thresholds = np.arange(0.50, 0.91, 0.05)
accuracies = []
uncertains = []

with torch.no_grad():
    for threshold in thresholds:
        all_preds = []
        all_labels = []
        uncertain_count = 0

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            for i in range(probs.size(0)):
                label = labels[i].item()
                prob_row = probs[i]
                top_confidence = prob_row.max().item()
                top_class = prob_row.argmax().item()

                all_labels.append(label)

                if top_confidence >= threshold:
                    all_preds.append(top_class)
                elif not exclude_uncertain:
                    all_preds.append(top_class)  # fallback
                else:
                    all_preds.append(None)
                    uncertain_count += 1

        # === FILTER OUT UNCERTAIN ===
        filtered_preds = [p for p in all_preds if p is not None]
        filtered_labels = [l for p, l in zip(all_preds, all_labels) if p is not None]

        correct = sum(p == l for p, l in zip(filtered_preds, filtered_labels))
        total = len(filtered_preds)
        acc = correct / total if total > 0 else 0

        accuracies.append(acc)
        uncertains.append(uncertain_count)

# === PLOT ACCURACY VS. THRESHOLD ===
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
plt.savefig("EfficientNetB3/accuracy_vs_threshold_acc.png")
plt.show()

# === PRINT THRESHOLD STATS ===
print("\n=== Threshold Summary ===")
for t, acc, u in zip(thresholds, accuracies, uncertains):
    print(f"Threshold {t:.2f} → Accuracy: {acc:.4f}, Uncertain: {u} ({u/len(test_dataset):.1%})")