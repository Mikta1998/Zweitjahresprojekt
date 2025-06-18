import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data_dir = "dataset/split"
model_path = "EfficientNetB3/train5/best_model_loss.pt"
num_classes = 7
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exclude_uncertain = True # Set to False to include uncertain

uncertainty_threshold = 0.80


transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names = test_dataset.classes


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

# === INFERENCE ===
all_preds = []
all_labels = []
all_pred_names = []
all_confidences = []
uncertain_count = 0

with torch.no_grad():
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

            if top_confidence >= uncertainty_threshold:
                all_preds.append(top_class)
                all_pred_names.append(class_names[top_class])
                all_confidences.append(top_confidence)
            elif not exclude_uncertain:
                # Use argmax anyway if uncertainty is allowed
                all_preds.append(top_class)
                all_pred_names.append(class_names[top_class] + " (uncertain)")
                all_confidences.append(top_confidence)
                uncertain_count += 1
            else:
                # Mark as uncertain
                all_preds.append(None)
                all_pred_names.append("Uncertain")
                all_confidences.append(top_confidence)
                uncertain_count += 1


filtered_preds = [p for p in all_preds if p is not None]
filtered_labels = [l for p, l in zip(all_preds, all_labels) if p is not None]


correct = sum(p == l for p, l in zip(filtered_preds, filtered_labels))
total = len(filtered_preds)
acc = correct / total if total > 0 else 0

print(f"\nAccuracy ({'excluding' if exclude_uncertain else 'including'} uncertain): {acc:.4f}")
print(f"Uncertain predictions: {uncertain_count} out of {len(all_labels)}")

print("\n=== Classification Report ===")
print(classification_report(filtered_labels, filtered_preds, target_names=class_names))


cm = confusion_matrix(filtered_labels, filtered_preds, labels=list(range(num_classes)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix" + (" (Filtered)" if exclude_uncertain else " (All Predictions)"))
plt.tight_layout()
plt.savefig("EfficientNetB3/confusion_matrix_80_loss.png")
plt.show()


df = pd.DataFrame({
    "True Label": [class_names[i] for i in all_labels],
    "Predicted Label": all_pred_names,
    "Confidence": all_confidences
})
df.to_csv("EfficientNetB3/predictions.csv", index=False)
print("Saved predictions to EfficientNetB3/predictions.csv")