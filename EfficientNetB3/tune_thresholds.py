import os
import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.metrics import f1_score
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

data_dir = "dataset/split"                            # Path to dataset 
model_path = "EfficientNetB3/best_model_acc.pt"       
num_classes = 7                                        
batch_size = 16                                       
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

transform = transforms.Compose([
    transforms.Resize((300, 300)),                    # Resize to EfficientNetB3 input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],       
                         [0.229, 0.224, 0.225])
])


val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
class_names = val_dataset.classes                

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

all_probs = []         # To store softmax probabilities
all_labels = []        # To store true class labels

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)       
        all_probs.append(probs.cpu())             
        all_labels.append(labels)

all_probs = torch.cat(all_probs).numpy()            
all_labels = torch.cat(all_labels).numpy()

thresholds = np.linspace(0.1, 0.9, 81)              # Test thresholds from 0.10 to 0.90 in steps of 0.01
best_thresholds = {}                                

for i, class_name in enumerate(class_names):
    best_f1 = 0
    best_thresh = 0.5
    true_binary = (all_labels == i).astype(int)   
    for t in thresholds:
        pred_binary = (all_probs[:, i] > t).astype(int)  # Binary prediction at threshold t
        f1 = f1_score(true_binary, pred_binary, zero_division=0)  # F1 score for this threshold
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    best_thresholds[class_name] = round(float(best_thresh), 4)    # Save best threshold
    print(f"{class_name}: Best Threshold = {best_thresh:.4f}, F1 = {best_f1:.4f}")

# Save best thresholds as JSON file
with open("EfficientNetB3/tuned_thresholds.json", "w") as f:
    json.dump(best_thresholds, f, indent=4)

print("\nSaved best thresholds to EfficientNetB3/tuned_thresholds.json")