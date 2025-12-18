import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from tools.data_loader import get_dataloaders
from tools.eval_utils import evaluate
from tools.runtime import get_device, set_seed


# =====================
# 0. Image dir
# =====================
IMAGE_DIR = "image"
os.makedirs(IMAGE_DIR, exist_ok=True)


# =====================
# 1. Runtime
# =====================
set_seed(42)
device = get_device()
print("Using device:", device)


# =====================
# 2. Data
# =====================
data_root = "Data"
batch_size = 32

train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
    data_root=data_root,
    batch_size=batch_size
)

num_classes = len(class_to_idx)
print("Classes:", class_to_idx)


# =====================
# 3. DenseNet-121
# =====================
weights = models.DenseNet121_Weights.IMAGENET1K_V1
model = models.densenet121(weights=weights)

in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, num_classes)

model = model.to(device)


# =====================
# 4. Loss & Optimizer
# =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# =====================
# 5. Training loop
# =====================
num_epochs = 15
best_val_acc = 0.0

train_accs = []
val_accs = []

for epoch in range(num_epochs):
    model.train()
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    val_loss, val_acc = evaluate(model, val_loader, device)

    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(
        f"[DenseNet-121] Epoch [{epoch+1}/{num_epochs}] "
        f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_densenet121.pth")


# =====================
# 6. Accuracy curve (save)
# =====================
epochs = list(range(1, num_epochs + 1))

plt.figure()
plt.plot(epochs, train_accs, marker='o', label="Train Acc")
plt.plot(epochs, val_accs, marker='o', label="Val Acc")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(epochs)
plt.title("DenseNet-121")

for x, y in zip(epochs, train_accs):
    plt.text(x, y, f"{y:.3f}", ha='center', va='bottom', fontsize=8)

for x, y in zip(epochs, val_accs):
    plt.text(x, y, f"{y:.3f}", ha='center', va='top', fontsize=8)

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(IMAGE_DIR, "densenet121_accuracy.png"),
    dpi=300
)
plt.close()


# =====================
# 7. Final test
# =====================
model.load_state_dict(
    torch.load("best_densenet121.pth", map_location=device)
)
model = model.to(device)

test_loss, test_acc = evaluate(model, test_loader, device)
print(f"DenseNet-121 Test Acc: {test_acc:.4f}")


# =====================
# 8. Confusion matrix
# =====================
def get_all_preds(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


y_true, y_pred = get_all_preds(model, test_loader)

cm = confusion_matrix(y_true, y_pred)

idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(num_classes)]

plt.figure()
plt.imshow(cm)
plt.title("DenseNet-121 Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)

plt.tight_layout()

plt.savefig(
    os.path.join(IMAGE_DIR, "densenet121_confusion_matrix.png"),
    dpi=300
)
plt.close()
