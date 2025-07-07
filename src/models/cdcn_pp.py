import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d_cd(nn.Conv2d):
    """Central Difference Convolutional 2D layer as proposed in CDCN++.
    When theta == 0 this layer behaves as a normal Conv2d.
    The central difference term is implemented by first computing a depthwise
    convolution with kernel weights summed over spatial dimensions (equivalent
    to taking the center pixel contribution) and subtracting it scaled by theta.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dilation: int = 1,
                 groups: int = 1, bias: bool = True, theta: float = 0.7):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias)
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard convolution
        out_normal = F.conv2d(x, self.weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)
        if self.theta == 0:
            return out_normal

        # Central difference convolution term
        kernel_diff = self.weight.sum(dim=(2, 3), keepdim=True)  # shape: (out_ch, in_ch/groups, 1, 1)
        out_diff = F.conv2d(x, kernel_diff, None, self.stride, 0, self.dilation, self.groups)
        return out_normal - self.theta * out_diff


class CDCNpp(nn.Module):
    """A lightweight CDCN++ backbone producing a 256-dimensional texture weight vector.

    The architecture here is a simplified variant suitable for small datasets.
    It contains three convolutional stages followed by global average pooling.
    The CDC weight vector is produced via a linear layer with sigmoid activation.
    Another small MLP head predicts class logits for the auxiliary loss (L1).
    """

    def __init__(self, num_classes: int = 2, theta: float = 0.7):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1
            Conv2d_cd(3, 64, kernel_size=3, padding=1, theta=theta),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Stage 2
            Conv2d_cd(64, 128, kernel_size=3, padding=1, theta=theta),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Stage 3
            Conv2d_cd(128, 256, kernel_size=3, padding=1, theta=theta),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # Produce 256-D weight vector (0-1 range)
        self.fc_weight = nn.Linear(256, 256)
        self.sigmoid = nn.Sigmoid()

        # Auxiliary classification head
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor):
        b = x.size(0)
        feat = self.features(x).view(b, -1)  # (B, 256)
        weight_vec = self.sigmoid(self.fc_weight(feat))  # (B, 256)
        logits = self.head(weight_vec)
        return weight_vec, logits
