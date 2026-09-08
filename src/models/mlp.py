import copy
import torch
from torch import nn
from torch.func import functional_call


class MLP(nn.Module):
    def __init__(self, dimension=16, width=256, depth=4, classes=4):
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(dimension if i == 0 else width, width)
                                     for i in range(depth)])
        self.head = nn.Linear(width, classes)
        for layer in self.hidden:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.head.weight, std=1/width**.5)
        nn.init.zeros_(self.head.bias)

    def representations(self, x):
        values = []
        for layer in self.hidden:
            x = torch.relu(layer(x))
            values.append(x)
        return values

    def forward(self, x):
        return self.head(self.representations(x)[-1])


class ScaledModel(nn.Module):
    def __init__(self, network, gamma=1., centered=True):
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.network = network
        self.gamma = gamma
        self.initial = copy.deepcopy(network).requires_grad_(False) if centered else None
        if self.initial is not None:
            self.initial.eval()
        self.has_batchnorm = any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in network.modules())

    def train(self, mode=True):
        super().train(mode)
        if self.initial is not None:
            self.initial.eval()
        return self

    def forward(self, x):
        if self.initial is not None and self.has_batchnorm and self.network.training:
            # Match training-mode batch statistics without changing reference
            # buffers. Otherwise centered BN models have nonzero initial logits.
            self.initial.train(True)
            try:
                buffers = {k:v.clone() for k,v in self.initial.named_buffers()}
                initial = functional_call(self.initial, (dict(self.initial.named_parameters()), buffers), (x,))
            finally:
                self.initial.eval()
        else:
            initial = self.initial(x) if self.initial is not None else 0
        return (self.network(x)-initial)/self.gamma
