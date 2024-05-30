import torch
import torchvision
from PIL import Image
import random

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, transform, test_transform, download=True, subset_ratio=1.0):
        super(CIFAR10, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False
        
        if subset_ratio < 1.0:
            num_samples = int(len(self.data) * subset_ratio)
            indices = random.sample(range(len(self.data)), num_samples)
            self.data = self.data[indices]
            self.targets = [self.targets[i] for i in indices]

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)            
        else:
            if self.transform is not None:
                img = self.transform(img)

        return img, target


class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, train, transform, test_transform, download=True, subset_ratio=1.0):
        super(CIFAR100, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False
        
        if subset_ratio < 1.0:
            num_samples = int(len(self.data) * subset_ratio)
            indices = random.sample(range(len(self.data)), num_samples)
            self.data = self.data[indices]
            self.targets = [self.targets[i] for i in indices]

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)            
        else:
            if self.transform is not None:
                img = self.transform(img)

        return img, target


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train, transform, test_transform, download=True, subset_ratio=1.0):
        super(MNIST, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False
        
        if subset_ratio < 1.0:
            num_samples = int(len(self.data) * subset_ratio)
            indices = random.sample(range(len(self.data)), num_samples)
            self.data = self.data[indices]
            self.targets = self.targets[indices]

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        
        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)            
        else:
            if self.transform is not None:
                img = self.transform(img)

        return img, target


class SVHN(torchvision.datasets.SVHN):
    def __init__(self, root, split, transform, test_transform, download=True, subset_ratio=1.0):
        super(SVHN, self).__init__(root, split, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False
        
        if subset_ratio < 1.0:
            num_samples = int(len(self.data) * subset_ratio)
            indices = random.sample(range(len(self.data)), num_samples)
            self.data = self.data[indices]
            self.labels = [self.labels[i] for i in indices]

    def __getitem__(self, index: int):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)
        
        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)            
        else:
            if self.transform is not None:
                img = self.transform(img)

        return img, target
