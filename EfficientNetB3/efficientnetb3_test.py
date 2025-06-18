import os
import shutil
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image

data_dir = "dataset/split"                                # Directory with test data
model_path = "EfficientNetB3/train6/best_model_loss.pt"   # Path to the trained model weights
num_classes = 7                                           # Total number of skin lesion classes
batch_size = 8                                            # Batch size lowered due to GPU memory issues
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

exclude_uncertain = False                                 # Whether to skip uncertain predictions
uncertainty_threshold = 0.80                              # Softmax threshold for confidence filtering

# Output directories for saving CAMs and uncertain cases
cam_dir = "EfficientNetB3/cam_visuals"
uncertain_dir = "EfficientNetB3/uncertain_predictions"
os.makedirs(cam_dir, exist_ok=True)
if os.path.exists(uncertain_dir):
    shutil.rmtree(uncertain_dir)
os.makedirs(uncertain_dir)

transform = transforms.Compose([
    transforms.Resize((300, 300)),                    
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],        
                         [0.229, 0.224, 0.225])
])

# Load dataset and labels
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names = test_dataset.classes                      
image_paths = [s[0] for s in test_dataset.samples]       

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

# Initialize GradCAM extractor
cam_extractor = GradCAM(model, target_layer="features.8")  # Use last conv layer

all_preds = []
all_labels = []
all_pred_names = []
all_confidences = []
uncertain_count = 0

for batch_idx, (inputs, labels) in enumerate(test_loader):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    probs = torch.softmax(outputs, dim=1)

    for i in range(probs.size(0)):
        label = labels[i].item()
        prob_row = probs[i]
        top_confidence = prob_row.max().item()
        top_class = prob_row.argmax().item()
        index = batch_idx * batch_size + i
        original_path = image_paths[index]

        all_labels.append(label)

        if top_confidence >= uncertainty_threshold:
            # Confident prediction then generate GradCAM
            all_preds.append(top_class)
            all_pred_names.append(class_names[top_class])
            all_confidences.append(top_confidence)

            original_img = Image.open(original_path).convert("RGB").resize((300, 300))

            cam = cam_extractor(class_idx=top_class, scores=outputs[i].unsqueeze(0), retain_graph=True)[0]
            cam = cam.mean(dim=0).unsqueeze(0)

            normalized_tensor = inputs[i].cpu()
            input_for_heatmap = to_pil_image(normalized_tensor.clamp(0, 1))
            heatmap = overlay_mask(input_for_heatmap, to_pil_image(cam.cpu(), mode='F'), alpha=0.5)
            heatmap = heatmap.resize((300, 300))

            combined = Image.new("RGB", (600, 300))
            combined.paste(original_img, (0, 0))
            combined.paste(heatmap, (300, 0))
            combined.save(f"{cam_dir}/sample_{batch_idx}_{i}_{class_names[top_class]}.png")

        else:
            # Handle uncertain predictions (record or skip based on setting)
            all_preds.append(top_class if not exclude_uncertain else None)
            all_pred_names.append(class_names[top_class] + " (uncertain)" if not exclude_uncertain else "Uncertain")
            all_confidences.append(top_confidence)
            uncertain_count += 1
            shutil.copy2(original_path, os.path.join(uncertain_dir, os.path.basename(original_path)))

filtered_preds = [p for p in all_preds if p is not None]
filtered_labels = [l for p, l in zip(all_preds, all_labels) if p is not None]

correct = sum(p == l for p, l in zip(filtered_preds, filtered_labels))
total = len(filtered_preds)
acc = correct / total if total > 0 else 0

print(f"\nAccuracy ({'excluding' if exclude_uncertain else 'including'} uncertain): {acc:.4f}")
print(f"Uncertain predictions: {uncertain_count} out of {len(all_labels)}")

print("\n=== Classification Report ===")
print(classification_report(filtered_labels, filtered_preds, target_names=class_names))

# === Save classification report as PNG ===
report_dict = classification_report(filtered_labels, filtered_preds, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(2)

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
table = ax.table(cellText=report_df.values,
                 colLabels=report_df.columns,
                 rowLabels=report_df.index,
                 loc='center',
                 cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.title("Classification Report", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("EfficientNetB3/classification_report.png", dpi=300)
plt.close()

# Confusion Matrix + Class Descriptions
cm = confusion_matrix(filtered_labels, filtered_preds, labels=list(range(num_classes)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation=45)

class_descriptions = {
    "akiec": "AKIEC: (Aktinische Keratose)",
    "bcc":   "BCC: (Basalzellkarzinom)",
    "bkl":   "BKL: (Benigne keratosenartige Läsionen)",
    "df":    "DF: (Dermatofibrom)",
    "mel":   "MEL: (Melanom)",
    "nv":    "NV: (Melanozytäre Nävi)",
    "vasc":  "VASC: (Vaskuläre Läsionen)"
}

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation=45, ax=ax, colorbar=True)

description_text = "\n".join([class_descriptions[k] for k in class_names])
plt.gcf().text(-0.4, 0.5, description_text, fontsize=10, va='center')

plt.title("Confusion Matrix (Filtered)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("EfficientNetB3/confusion_matrix_with_labels_left.png", dpi=300, bbox_inches='tight')
plt.show()

# Save predictions to CSV 
df = pd.DataFrame({
    "True Label": [class_names[i] for i in all_labels],     # Actual label
    "Predicted Label": all_pred_names,                      # Predicted or uncertain
    "Confidence": all_confidences                           # Max softmax value
})
df.to_csv("EfficientNetB3/predictions.csv", index=False)
print("Saved predictions to EfficientNetB3/predictions.csv")
print("Uncertain images saved to:", uncertain_dir)