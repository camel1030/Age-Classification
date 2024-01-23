import torch
import torch.nn as nn
import torch.nn.functional as F


# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#
#         # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1,
#                       bias=False),
#             nn.BatchNorm2d(out_channels * BasicBlock.expansion),
#         )
#
#         # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
#         self.shortcut = nn.Sequential()
#
#         self.relu = nn.ReLU()
#
#         # projection mapping using 1x1conv
#         if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * BasicBlock.expansion)
#             )
#
#     def forward(self, x):
#         x = self.residual_function(x) + self.shortcut(x)
#         x = self.relu(x)
#         return x
#
# class BottleNeck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(out_channels * BottleNeck.expansion),
#         )
#
#         self.shortcut = nn.Sequential()
#
#         self.relu = nn.ReLU()
#
#         if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * BottleNeck.expansion)
#             )
#
#     def forward(self, x):
#         x = self.residual_function(x) + self.shortcut(x)
#         x = self.relu(x)
#         return x
#
# class ResNet(nn.Module):
#     def __init__(self, block, num_block, num_classes=10, init_weights=True):
#         super().__init__()
#
#         self.in_channels=64
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )
#
#         self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
#         self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
#         self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
#         self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
#
#         self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         # weights inittialization
#         if init_weights:
#             self._initialize_weights()
#
#     def _make_layer(self, block, out_channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channels, out_channels, stride))
#             self.in_channels = out_channels * block.expansion
#
#         return nn.Sequential(*layers)
#
#     def forward(self,x):
#         output = self.conv1(x)
#         output = self.conv2_x(output)
#         x = self.conv3_x(output)
#         x = self.conv4_x(x)
#         x = self.conv5_x(x)
#         x = self.avg_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
#
#     # define weight initialization function
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
# def resnet18():
#     return ResNet(BasicBlock, [2,2,2,2])
#
# def resnet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])
#
# def resnet50():
#     return ResNet(BottleNeck, [3,4,6,3])
#
# def resnet101():
#     return ResNet(BottleNeck, [3, 4, 23, 3])
#
# def resnet152():
#     return ResNet(BottleNeck, [3, 8, 36, 3])

##########################
# MODEL
##########################


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):    #resnet34(Basicblock, [3,4,6,3], )

    def __init__(self, block, layers, num_classes, grayscale):
        self.num_classes = num_classes
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,  #nn.Conv2d: 이미지 등과 같은 2D 데이터의 특징 추출(64,64*1,1,)
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)       #입력 채널 = 64
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=2)
        self.fc = nn.Linear(2048 * block.expansion, 1, bias=False)
        #self.linear_1_bias = nn.Parameter(torch.zeros(self.num_classes-1).float())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1): #block: Basicblock  planes:
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
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)   #(16,2048)
        logits = self.fc(x)
        # logits = logits + self.linear_1_bias
        # probas = torch.sigmoid(logits)
        return logits  #, probas

def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock,
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model
