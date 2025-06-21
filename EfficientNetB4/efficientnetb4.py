import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from collections import Counter

# === Settings ===
data_dir = "dataset/split"
num_classes = 7
batch_size = 16
learning_rate = 1e-4
num_epochs = 50
patience = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transforms for EfficientNet‑B4 ===
transform = transforms.Compose([
    transforms.Resize(384, transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

datasets_map = {split: datasets.ImageFolder(os.path.join(data_dir, split), transform)
                for split in ['train','val','test']}
dataloaders = {split: DataLoader(datasets_map[split], batch_size=batch_size, shuffle=(split=='train'))
               for split in ['train','val','test']}
class_names = datasets_map['train'].classes

# === Class balance check & loss weighting ===
counts = Counter(label for _, label in datasets_map['train'].samples)
sample_counts = list(counts.values())
is_balanced = max(sample_counts) - min(sample_counts) < 0.1 * max(sample_counts)

if is_balanced:
    print("Balanced dataset — using unweighted loss.")
    criterion = nn.CrossEntropyLoss()
else:
    print("Imbalanced dataset — applying class weights.")
    class_counts = torch.tensor(sample_counts, dtype=torch.float32)
    weights = (1.0 / class_counts)
    weights = weights / weights.sum() * len(class_counts)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

# === Define EfficientNet‑B4 model ===
class B4SkinLesion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        bm = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        self.features = bm.features
        self.pool = bm.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(bm.classifier[1].in_features, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(torch.flatten(x, 1))

model = B4SkinLesion(num_classes).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)
writer = SummaryWriter(log_dir='runs/efficientnetb4_skin_lesion')

os.makedirs("EfficientNetB4", exist_ok=True)
train_loss_curve, val_loss_curve = [], []
train_acc_curve, val_acc_curve = [], []

best_val_loss, best_val_acc = float('inf'), 0.0
patience_ctr = 0

# === Training Loop ===
for epoch in range(num_epochs):
    model.train()
    tl, tc, tt = 0.0, 0, 0
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        tl += loss.item() * labels.size(0)
        tc += (outputs.argmax(1) == labels).sum().item()
        tt += labels.size(0)

    train_loss = tl / tt
    train_acc = tc / tt

    model.eval()
    vl, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            vl += loss.item() * labels.size(0)
            vc += (outputs.argmax(1) == labels).sum().item()
            vt += labels.size(0)

    val_loss = vl / vt
    val_acc = vc / vt

    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Accuracy/Train", train_acc, epoch)
    writer.add_scalar("Loss/Validation", val_loss, epoch)
    writer.add_scalar("Accuracy/Validation", val_acc, epoch)

    train_loss_curve.append(train_loss)
    val_loss_curve.append(val_loss)
    train_acc_curve.append(train_acc)
    val_acc_curve.append(val_acc)

    print(f"Epoch {epoch+1} | Train {train_loss:.4f}, Acc {train_acc:.4f} | Val {val_loss:.4f}, Acc {val_acc:.4f}")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        torch.save(model.state_dict(), "EfficientNetB4/best_model_loss.pt")
        best_val_loss = val_loss
        patience_ctr = 0
        print("Saved new best (by loss).")
    else:
        patience_ctr += 1

    if val_acc > best_val_acc:
        torch.save(model.state_dict(), "EfficientNetB4/best_model_acc.pt")
        best_val_acc = val_acc
        print("Saved new best (by accuracy).")

    if patience_ctr >= patience:
        print("Early stopping.")
        break

writer.close()

# === Plot Curves with Grid ===
epochs = range(1, len(train_loss_curve) + 1)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs, train_loss_curve, label='Train Loss')
plt.plot(epochs, val_loss_curve, label='Validation Loss')
plt.title("Training Curve")
plt.xlabel("Epoch"), plt.ylabel("Loss")
plt.legend(), plt.grid(True)

plt.subplot(1,2,2)
plt.plot(epochs, train_acc_curve, label='Train Acc')
plt.plot(epochs, val_acc_curve, label='Val Acc')
plt.title("Accuracy Curve")
plt.xlabel("Epoch"), plt.ylabel("Accuracy")
plt.legend(), plt.grid(True)

plt.tight_layout()
plt.savefig("EfficientNetB4/training_curves.png", dpi=300)
plt.show()