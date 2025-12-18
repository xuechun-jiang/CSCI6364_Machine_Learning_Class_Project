import torch
import torch.nn as nn
import numpy as np
from torchvision import models

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from tools.data_loader import get_dataloaders
from tools.runtime import get_device, set_seed

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
# 3. Feature extractor factory
# =====================
def get_feature_extractor(model_name):
    if model_name == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        extractor = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
            model.avgpool, nn.Flatten()
        )

    elif model_name == "resnet34":
        model = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1
        )
        extractor = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
            model.avgpool, nn.Flatten()
        )

    elif model_name == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )
        extractor = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
            model.avgpool, nn.Flatten()
        )

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        extractor = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    elif model_name == "densenet121":
        model = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1
        )
        extractor = nn.Sequential(
            model.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    extractor = extractor.to(device)
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False

    return extractor

# =====================
# 4. Feature extraction
# =====================
def extract_features(loader, extractor):
    features = []
    labels = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            feats = extractor(images)

            features.append(feats.cpu().numpy())
            labels.append(targets.numpy())

    return np.concatenate(features), np.concatenate(labels)

# =====================
# 5. SVM training & evaluation
# =====================
def run_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    svm = SVC(
        kernel="rbf",     # 可改成 "linear" 做对照
        C=10,
        gamma="scale"
    )

    svm.fit(X_train, y_train)

    val_acc = accuracy_score(y_val, svm.predict(X_val))
    test_acc = accuracy_score(y_test, svm.predict(X_test))

    return val_acc, test_acc

# =====================
# 6. Main experiment loop
# =====================
models_to_test = [
    "resnet18",
    "resnet34",
    "resnet50",
    "efficientnet_b0",
    "densenet121"
]

results = {}

for name in models_to_test:
    print(f"\n=== {name} + SVM ===")

    extractor = get_feature_extractor(name)

    print("Extracting train features...")
    X_train, y_train = extract_features(train_loader, extractor)

    print("Extracting val features...")
    X_val, y_val = extract_features(val_loader, extractor)

    print("Extracting test features...")
    X_test, y_test = extract_features(test_loader, extractor)

    print("Training SVM...")
    val_acc, test_acc = run_svm(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    results[name] = (val_acc, test_acc)
    print(
        f"{name} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}"
    )

# =====================
# 7. Summary
# =====================
print("\n=== Final Summary ===")
for name, (val_acc, test_acc) in results.items():
    print(
        f"{name:15s} | Val: {val_acc:.4f} | Test: {test_acc:.4f}"
    )
