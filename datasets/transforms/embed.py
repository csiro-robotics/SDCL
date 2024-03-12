import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models

import torch.utils.data as data


def r18_extractor() -> nn.Module:
    """Create a feature extractor from a pre-trained ResNet18"""
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()

    """ResNet18 feature extractor using `model.eval()`, `torch.no_grad()`, and 
    requires_grad=False` performs much better 5% -> 78%. Forcing it into eval
    mode in particular is important. Scaling the input to 224x224 is also
    important because the model was trained on 224x224 images.
    """

    @torch.no_grad()
    def _forward(x: Tensor) -> Tensor:
        model.eval()
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return model._forward_impl(x).detach()

    model.forward = _forward
    for p in model.parameters():
        p.requires_grad = False
    return model


R18_EXTRACTOR = r18_extractor().cuda()
"""Singleton ResNet18 feature extractor"""


class BatchTransformDataLoader:
    def __init__(self, dataloader, batch_transform):
        self.dataloader = dataloader
        self.batch_transform = batch_transform

    def __iter__(self):
        for batch in self.dataloader:
            batch = [x.cuda() for x in batch]
            yield self.batch_transform(*batch)

    @property
    def dataset(self):
        return self.dataloader.dataset

    def __len__(self):
        return len(self.dataloader)
