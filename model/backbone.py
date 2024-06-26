from torch import nn
import torchvision


class ResNetBackbone(nn.Module):
    def __init__(self, net):
        super(ResNetBackbone, self).__init__()
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = self.layer1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1, x2, x3, x4, x5

    def initialize(self):
        pass


def resnet_backbone(name, **kwargs):
    resnets = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    if name in resnets:
        cnn = getattr(torchvision.models, name)(**kwargs)
        return ResNetBackbone(cnn)
    else:
        raise NameError('unrecognized backbone "%s"' % name)
