from typing import Tuple

import torch
from sklearn.metrics import accuracy_score, roc_auc_score


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds_label = preds.argmax(dim=1).detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    return accuracy_score(targets_np, preds_label)


def auc(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Compute AUC using sklearn. Handles binary and multi-class by OVR."""
    preds_prob = torch.softmax(preds, dim=1).detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    if num_classes == 2:
        return roc_auc_score(targets_np, preds_prob[:, 1])
    else:
        return roc_auc_score(targets_np, preds_prob, multi_class='ovr')
