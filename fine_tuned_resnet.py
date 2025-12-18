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


IMAGE_DIR = "image"
os.makedirs(IMAGE_DIR, exist_ok=True)

set_seed(42)
device = get_device()
print("Using device:", device)

data_root = "Data"
batch_size = 32

train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
    data_root=data_root,
    batch_size=batch_size
)

num_classes = len(class_to_idx)
print("Classes:", class_to_idx)

def get_resnet(model_name, num_classes):
    if model_name == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
    elif model_name == "resnet34":
        model = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1
        )
    elif model_name == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

def train_resnet(model, train_loader, val_loader, model_name, epochs=10):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
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
            f"[{model_name}] Epoch [{epoch+1}/{epochs}] "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{model_name}.pth")

    return best_val_acc, train_accs, val_accs

def plot_accuracy_curve(train_accs, val_accs, model_name):
    epochs = list(range(1, len(train_accs) + 1))

    plt.figure()
    plt.plot(epochs, train_accs, marker='o', label="Train Acc")
    plt.plot(epochs, val_accs, marker='o', label="Val Acc")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs)
    plt.title(model_name)

    for x, y in zip(epochs, train_accs):
        plt.text(x, y, f"{y:.3f}", ha='center', va='bottom', fontsize=8)

    for x, y in zip(epochs, val_accs):
        plt.text(x, y, f"{y:.3f}", ha='center', va='top', fontsize=8)

    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(
        IMAGE_DIR, f"{model_name}_accuracy.png"
    )
    plt.savefig(save_path, dpi=300)
    plt.close()


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


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)

    plt.tight_layout()

    save_path = os.path.join(
        IMAGE_DIR, f"{model_name}_confusion_matrix.png"
    )
    plt.savefig(save_path, dpi=300)
    plt.close()


resnet_variants = ["resnet18", "resnet34", "resnet50"]
num_epochs = 15

results = {}

for name in resnet_variants:
    print(f"\n=== Training {name} ===")
    model = get_resnet(name, num_classes)

    best_val_acc, train_accs, val_accs = train_resnet(
        model,
        train_loader,
        val_loader,
        model_name=name,
        epochs=num_epochs
    )

    results[name] = best_val_acc

    plot_accuracy_curve(train_accs, val_accs, name)


print("\n=== Final Test Evaluation ===")

idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(num_classes)]

for name in resnet_variants:
    model = get_resnet(name, num_classes)
    model.load_state_dict(
        torch.load(f"best_{name}.pth", map_location=device)
    )
    model = model.to(device)

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"{name} Test Acc: {test_acc:.4f}")

    y_true, y_pred = get_all_preds(model, test_loader)
    plot_confusion_matrix(y_true, y_pred, class_names, name)
