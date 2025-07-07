from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_malaria_loaders(root: str, img_size: int = 32, batch_size: int = 128,
                        val_split: float = 0.1, test_split: float = 0.1,
                        num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/val/test dataloaders for the Malaria cell images dataset.

    The directory structure under `root` is expected as:
        root/
            Parasitized/
                *.png
            Uninfected/
                *.png
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Malaria dataset directory not found: {root}")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(root=root, transform=transform)
    total_len = len(full_dataset)
    val_len = int(total_len * val_split)
    test_len = int(total_len * test_split)
    train_len = total_len - val_len - test_len
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
