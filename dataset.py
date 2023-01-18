import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch


# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class Cifar10Mod(torchvision.datasets.CIFAR10):
    def __init__(self, root: str,train, hr_shape):
        super().__init__(root, train=train)
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height // 4), transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __getitem__(self, item):
        image, _ = super().__getitem__(item)
        img_lr = self.lr_transform(image)
        img_hr = self.hr_transform(image)

        return {"lr": img_lr, "hr": img_hr}
