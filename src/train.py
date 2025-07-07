import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.twt import TwT
from datasets.malaria_dataset import get_malaria_loaders
from datasets.blood_mnist_dataset import get_bloodmnist_loaders
from utils.metrics import accuracy, auc


def train_one_epoch(model, loader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss, running_acc = 0.0, 0.0
    for images, labels in tqdm(loader, desc='Train', leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits_tex, logits_spa = model(images)
        loss_tex = criterion(logits_tex, labels)
        loss_spa = criterion(logits_spa, labels)
        loss = loss_tex + loss_spa
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_acc += accuracy(logits_spa, labels) * images.size(0)  # use spatial branch for acc
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, total_acc, total_auc = 0.0, 0.0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Eval', leave=False):
            images, labels = images.to(device), labels.to(device)
            logits_tex, logits_spa = model(images)
            loss_tex = criterion(logits_tex, labels)
            loss_spa = criterion(logits_spa, labels)
            loss = loss_tex + loss_spa
            total_loss += loss.item() * images.size(0)
            total_acc += accuracy(logits_spa, labels) * images.size(0)
            total_auc += auc(logits_spa, labels, num_classes) * images.size(0)

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = total_acc / len(loader.dataset)
    avg_auc = total_auc / len(loader.dataset)
    return avg_loss, avg_acc, avg_auc


def main():
    parser = argparse.ArgumentParser(description='Train Texture Weighted Transformer (TwT)')
    parser.add_argument('--dataset', choices=['malaria', 'bloodmnist'], required=True)
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--theta', type=float, default=0.7)
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'malaria':
        train_loader, val_loader, test_loader = get_malaria_loaders(
            root=args.data_root, batch_size=args.batch_size)
        num_classes = 2
        in_channels = 3
    else:
        train_loader, val_loader, test_loader, num_classes, in_channels = get_bloodmnist_loaders(
            download_root=args.data_root, batch_size=args.batch_size)

    model = TwT(num_classes=num_classes, theta=args.theta)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, num_classes)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device, num_classes)
        print(f'Epoch {epoch}: Train Loss {train_loss:.4f} Acc {train_acc:.4f} | '
              f'Val Loss {val_loss:.4f} Acc {val_acc:.4f} AUC {val_auc:.4f}')
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_dir / f'best_model_{args.dataset}.pth')

    # Final test evaluation
    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion, device, num_classes)
    print(f'Test: Loss {test_loss:.4f} Acc {test_acc:.4f} AUC {test_auc:.4f}')


if __name__ == '__main__':
    main()
