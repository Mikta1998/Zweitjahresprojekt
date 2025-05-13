import os
import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Config
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 40
PATIENCE = 5  # early stopping patience
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = 'dataset/split_nested/train_balanced'
VAL_DIR = 'dataset/split_nested/val'
TEST_DIR = 'dataset/split_nested/test'

plt.ion()

# Transform
common_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasets & Loaders
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=common_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=common_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=common_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model: switch to smaller and more efficient model
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES, drop_rate=0.3)
model.to(DEVICE)

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logpt = -self.ce(input, target)
        pt = torch.exp(logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        return -loss

criterion = FocalLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

train_losses, val_losses, val_aucs, val_accs = [], [], [], []
best_auc = 0
no_improve_epochs = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

    for inputs, labels in loop:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            loss = criterion(outputs, labels)
            val_loss += loss.item()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = correct / total
    val_auc = roc_auc_score(y_true, y_scores)

    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)
    val_aucs.append(val_auc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc:.4f} | "
          f"Val AUC: {val_auc:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.1e}")

    if val_auc > best_auc:
        best_auc = val_auc
        no_improve_epochs = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print("✅ Saved new best model!")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= PATIENCE:
            print(f"⛔ Early stopping at epoch {epoch+1}")
            break

    scheduler.step()

    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Val Acc')
    plt.plot(val_aucs, label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.pause(0.01)

plt.ioff()
plt.show()

# Evaluation

def evaluate(loader, name):
    model.eval()
    model.load_state_dict(torch.load('best_model.pth'))
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())

    print(f"\n{name} Classification Report")
    print(classification_report(y_true, y_pred, target_names=train_dataset.classes))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

evaluate(val_loader, "Validation")
evaluate(test_loader, "Test")
