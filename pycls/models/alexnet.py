import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Any


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10, use_dropout=False):
        super(AlexNet, self).__init__()
        self.use_dropout = use_dropout
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.drop1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        # Adjusted for 32x32 input size, after pooling the size will be 16x16
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        if self.use_dropout:
            x = self.drop1(x)
        # print(f"Shape before flatten: {x.shape}")  # Debug print to verify shape
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        if self.use_dropout:
            x = self.drop2(x)
        x = self.fc2(x)
        return x


# class AlexNet(nn.Module):
#     '''
#     AlexNet modified (features) for CIFAR10. Source: https://github.com/icpm/pytorch-cifar10/blob/master/models/AlexNet.py. 
#     '''
#     def __init__(self, num_classes: int = 1000, use_dropout=False) -> None:
#         super(AlexNet, self).__init__()
#         self.use_dropout = use_dropout
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(64, 192, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#         )
#         # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         self.fc_block = nn.Sequential(
#             nn.Linear(256 * 2 * 2, 4096, bias=False),
#             nn.BatchNorm1d(4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, 4096, bias=False),
#             nn.BatchNorm1d(4096),
#             nn.ReLU(inplace=True),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(4096, num_classes),
#         )
#         self.penultimate_active = False
#         self.drop = nn.Dropout(p=0.5)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         # x = self.avgpool(x)
#         z = torch.flatten(x, 1)
#         if self.use_dropout:
#             x = self.drop(x)
#         z = self.fc_block(z)
#         x = self.classifier(z)
#         if self.penultimate_active:
#             return z, x
#         return x


def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model