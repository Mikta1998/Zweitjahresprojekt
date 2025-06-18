import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from collections import Counter

data_dir = "dataset/split"                 
num_classes = 7                            # Number of output classes
batch_size = 16                           
learning_rate = 1e-4                       # Learning rate for optimizer
num_epochs = 50                            # Maximum number of training epochs
patience = 15                              # Early stopping patience
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Preprocessing/Normalization
base_transform = transforms.Compose([
    transforms.Resize((300, 300)),         # 300x300 input for EfficientNetB3
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # Standard ImageNet normalization
                         [0.229, 0.224, 0.225])
])

datasets_map = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), base_transform),
    'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), base_transform),
    'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), base_transform)
}
dataloaders = {
    split: DataLoader(datasets_map[split], batch_size=batch_size, shuffle=(split == 'train'))
    for split in ['train', 'val', 'test']
}
class_names = datasets_map['train'].classes  # Class labels

# Check for class imbalance in training set
label_counts = Counter()
for _, label in datasets_map['train']:
    label_counts[label] += 1
sample_counts = list(label_counts.values())
is_balanced = max(sample_counts) - min(sample_counts) < 0.1 * max(sample_counts)


class EfficientNetB3SkinLesion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)  # Pretrained EfficientNet
        self.features = base_model.features
        self.pooling = base_model.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),               # Deactivate random neurons
            nn.Linear(base_model.classifier[1].in_features, num_classes)  # Output layer
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)                            # Flatten for the classifier
        return self.classifier(x)

model = EfficientNetB3SkinLesion(num_classes).to(device)

if is_balanced:
    print("Balanced dataset detected — using unweighted loss.")
    criterion = nn.CrossEntropyLoss()
else:
    print("Imbalanced dataset detected — using class weights.")
    class_counts = torch.tensor(sample_counts, dtype=torch.float32)
    weights = 1.0 / class_counts                           
    weights = weights / weights.sum() * len(class_counts) # Normalize weights
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)
writer = SummaryWriter(log_dir='runs/efficientnetb3_skin_lesion') 

# Tracks losses and accuracy across epochs
train_loss_curve = []
val_loss_curve = []
train_acc_curve = []
val_acc_curve = []

# Training Loop 
best_val_loss = float('inf')
best_val_acc = 0.0
patience_counter = 0
os.makedirs("EfficientNetB3", exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()                  
        outputs = model(inputs)                
        loss = criterion(outputs, labels)      # Compute loss
        loss.backward()                        # Backpropagation
        optimizer.step()                       # Optimizer update

        train_loss += loss.item() * inputs.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    # Validation Loop
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Accuracy/Train", train_acc, epoch)
    writer.add_scalar("Loss/Validation", val_loss, epoch)
    writer.add_scalar("Accuracy/Validation", val_acc, epoch)

    # Store metrics for plotting later
    train_loss_curve.append(train_loss)
    val_loss_curve.append(val_loss)
    train_acc_curve.append(train_acc)
    val_acc_curve.append(val_acc)

    # Print epoch progress 
    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    scheduler.step(val_loss)  # Learning rate scheduler reacts to val loss

    # Save model if it's the best so far (by validation loss) 
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), "EfficientNetB3/best_model_loss.pt")
        best_val_loss = val_loss
        print("Saved new best model (by val_loss)")
        patience_counter = 0
    else:
        patience_counter += 1

    # Also save by best accuracy 
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), "EfficientNetB3/best_model_acc.pt")
        best_val_acc = val_acc
        print("Saved new best model (by val_acc)")

    # Early stopping 
    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

writer.close()

# Plot training curves
epochs = range(1, len(train_loss_curve) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_curve, label='Train Loss')
plt.plot(epochs, val_loss_curve, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc_curve, label='Train Accuracy')
plt.plot(epochs, val_acc_curve, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.savefig("EfficientNetB3/training_curves.png")  # Save the figure to disk
plt.show()