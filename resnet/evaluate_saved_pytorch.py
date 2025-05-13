import os
import torch
import timm
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Config
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"

# Data paths
val_dir = 'dataset/split_nested/val'
test_dir = 'dataset/split_nested/test'

# ImageNet-normalized transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets and loaders
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES, drop_rate=0.3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Evaluation function
def evaluate_model(loader, dataset_name):
    y_true = []
    y_pred = []
    class_names = loader.dataset.classes

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Classification report
    print(f"\n{dataset_name} Classification Report")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title(f"{dataset_name} - Confusion Matrix")
    plt.savefig(f"{dataset_name.lower().replace(' ', '_')}_confusion_matrix.png", bbox_inches="tight")
    plt.show()

# Run evaluation
evaluate_model(val_loader, "Validation Set")
evaluate_model(test_loader, "Test Set")