import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Function

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_maps = self.layer4(x)

        x = self.avgpool(feature_maps)
        feature = x.view(x.size(0), -1)
        x = self.fc(feature)

        return feature, x, feature_maps


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, residual_transform=None, output_activation='relu', norm='batch'):
        super(ResNetBasicblock, self).__init__()
        self.norm = norm

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if norm == 'batch':
            self.bn_a = nn.BatchNorm2d(planes)
        elif norm == 'instance':
            self.bn_a = nn.InstanceNorm2d(planes)
        else:
            assert False, 'norm must be batch or instance'

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if norm == 'batch':
            self.bn_b = nn.BatchNorm2d(planes)
        elif norm == 'instance':
            self.bn_b = nn.InstanceNorm2d(planes)
        else:
            assert False, 'norm must be batch or instance'

        self.residual_transform = residual_transform
        self.output_activation = nn.ReLU() if output_activation == 'relu' else nn.Tanh()

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)
        # basicblock = F.leaky_relu(basicblock, 0.1, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.residual_transform is not None:
            residual = self.residual_transform(x)

        if residual.size()[1] > basicblock.size()[1]:
            residual = residual[:, :basicblock.size()[1], :, :]
        output = self.output_activation(residual + basicblock)
        return output

def init_params(m):
    """
    initialize a module's parameters
    if conv2d or convT2d, using he normalization
    if bn set weight to 1 and bias to 0
    :param m:
    :return:
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

class Resnet50WithDomainClassifier(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50WithDomainClassifier, self).__init__()
        self.layers = [3, 4, 6, 3]
        self.inplanes = 64
        self.num_classes = num_classes

        # fe = feature_extractor
        self.fe_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fe_bn1 = nn.BatchNorm2d(64)
        self.fe_relu = nn.ReLU(inplace=True)
        self.fe_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fe_layer1 = self._make_layer(Bottleneck, 64, self.layers[0])
        self.fe_layer2 = self._make_layer(Bottleneck, 128, self.layers[1], stride=2)
        self.fe_layer3 = self._make_layer(Bottleneck, 256, self.layers[2], stride=2)
        self.fe_layer4 = self._make_layer(Bottleneck, 512, self.layers[3], stride=2)
        self.fe_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # lc = label_classifier
        self.lc_fc = nn.Linear(512 * Bottleneck.expansion, self.num_classes)

        # vc = view_classifier
        self.vc_fc1 = nn.Linear(512 * Bottleneck.expansion, 512)
        self.vc_relu1 = nn.ReLU(inplace=True)
        self.vc_fc2 = nn.Linear(512, 2)

    def _make_layer(self, block, planes, blocks_num, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks_num):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, alpha=0):
        x = self.fe_conv1(x)
        x = self.fe_bn1(x)
        x = self.fe_relu(x)
        x = self.fe_maxpool(x)
        x = self.fe_layer1(x)
        x = self.fe_layer2(x)
        x = self.fe_layer3(x)
        x = self.fe_layer4(x)
        x = self.fe_avgpool(x)
        feature = x.view(x.size(0), -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        label = self.lc_fc(feature)
        view_x = self.vc_fc1(reverse_feature)
        view_x = self.vc_relu1(view_x)
        domain = self.vc_fc2(view_x)

        return feature, label, domain

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_outputs):
        output = grad_outputs.neg() * ctx.alpha

        return output, None