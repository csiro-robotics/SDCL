# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, Subset
from datasets.utils.subset import MammothSubset

from utils import create_if_not_exists


class ValidationDataset(MammothSubset):
    def __init__(self, dataset: Dataset, transform: Optional[nn.Module] = None):
        """
        Creates a dataset for validation.
        :param dataset: the dataset to be used
        :param transform: the transformation to be applied
        """
        super(ValidationDataset, self).__init__(dataset, [])
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.dataset[index][2]), self.dataset[index][1]


def get_train_val(
    train: Dataset, test_transform: nn.Module, dataset: str, val_perc: float = 0.1
):
    """
    Extract val_perc% of the training set as the validation set.
    :param train: training dataset
    :param test_transform: transformation of the test dataset
    :param dataset: dataset name
    :param val_perc: percentage of the training set to be extracted
    :return: the training set and the validation set
    """
    dataset_length = train.data.shape[0]
    directory = "datasets/val_permutations/"
    create_if_not_exists(directory)
    file_name = dataset + ".pt"
    if os.path.exists(directory + file_name):
        perm = torch.load(directory + file_name)
    else:
        perm = torch.randperm(dataset_length)
        torch.save(perm, directory + file_name)

    train_count = int((1 - val_perc) * dataset_length)
    train_indices = perm[:train_count]
    val_indices = perm[train_count:]

    test_dataset = ValidationDataset(
        train,
        val_indices,
        transform=test_transform,
    )

    return MammothSubset(train, train_indices), test_dataset
