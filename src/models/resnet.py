from torch import nn
from torchvision.models import resnet18


def cifar_resnet(classes=10):
    model = resnet18(weights=None, num_classes=classes)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


class CIFARResNet(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.backbone = cifar_resnet(classes)

    def representations(self, x):
        m = self.backbone
        x = m.relu(m.bn1(m.conv1(x)))
        values = []
        for layer in [m.layer1, m.layer2, m.layer3, m.layer4]:
            x = layer(x)
            values.append(m.avgpool(x).flatten(1))
        return values

    def forward(self, x):
        return self.backbone.fc(self.representations(x)[-1])
