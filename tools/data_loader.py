from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from tools.data_transforms import get_train_transform, get_eval_transform


def limit_per_class(dataset, max_per_class=200):
    class_counter = defaultdict(int)
    selected_indices = []

    for idx, (_, class_idx) in enumerate(dataset.samples):
        if class_counter[class_idx] < max_per_class:
            selected_indices.append(idx)
            class_counter[class_idx] += 1

    return Subset(dataset, selected_indices)


def get_dataloaders(data_root, batch_size=32, max_per_class=200):
    # -------- Train --------
    train_ds_full = datasets.ImageFolder(
        root=f"{data_root}/train",
        transform=get_train_transform()
    )

    # ✅ 只在 train 时限制每类 200 张
    train_ds = limit_per_class(train_ds_full, max_per_class)

    # -------- Validation --------
    val_ds = datasets.ImageFolder(
        root=f"{data_root}/valid",
        transform=get_eval_transform()
    )
    val_ds = limit_per_class(val_ds, 50)

    # -------- Test --------
    test_ds = datasets.ImageFolder(
        root=f"{data_root}/test",
        transform=get_eval_transform()
    )
    test_ds = limit_per_class(test_ds, 50)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader, train_ds_full.class_to_idx
