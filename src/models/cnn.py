import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, channels=1, classes=10, width=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width*2, 3, padding=1)
        self.projection = nn.Linear(width*2*6*6, 64)
        self.head = nn.Linear(64, classes)

    def representations(self, x):
        a = torch.relu(self.conv1(x))
        b = torch.relu(self.conv2(torch.nn.functional.avg_pool2d(a, 2)))
        c = torch.relu(self.projection(torch.nn.functional.adaptive_avg_pool2d(b, (6, 6)).flatten(1)))
        # Spatially retained pooled activations preserve position/orientation.
        return [torch.nn.functional.adaptive_avg_pool2d(a, (6, 6)).flatten(1),
                torch.nn.functional.adaptive_avg_pool2d(b, (6, 6)).flatten(1), c]

    def forward(self, x):
        return self.head(self.representations(x)[-1])
