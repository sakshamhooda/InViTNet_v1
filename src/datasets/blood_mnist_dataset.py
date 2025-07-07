from typing import Tuple

import torch
from medmnist import INFO, BloodMNIST
from torch.utils.data import DataLoader
from torchvision import transforms


def get_bloodmnist_loaders(download_root: str = './data', img_size: int = 28,
                           batch_size: int = 128, val_split: float = 0.1,
                           num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/val/test dataloaders for BloodMNIST using the medmnist API."""
    info = INFO['bloodmnist']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_dataset = BloodMNIST(split='train', transform=data_transform, download=True, root=download_root)
    test_dataset = BloodMNIST(split='test', transform=data_transform, download=True, root=download_root)

    # Create validation split from train
    total_train = len(train_dataset)
    val_len = int(total_train * val_split)
    train_len = total_train - val_len
    train_set, val_set = torch.utils.data.random_split(
        train_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, n_classes, n_channels
