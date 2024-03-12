# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from datasets.utils.continual_dataset import ContinualDataset
from torch.utils.data import Dataset, DataLoader, ConcatDataset


from datasets.utils.subset import MammothSubset


class MemoryDataset(Dataset):
    def __init__(
        self, data: torch.Tensor, targets: torch.Tensor, transform=lambda x: x
    ) -> None:
        super().__init__()
        self.data = data.float().detach()
        self.targets = targets.long().detach()
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return (
            self.transform(self.data[index]),
            int(self.targets[index]),
            self.data[index],
        )

    def __len__(self) -> int:
        return len(self.data)


def icarl_replay(self, dataset: ContinualDataset, val_set_split=0):
    """
    Merge the replay buffer with the current task data.
    Optionally split the replay buffer into a validation set.

    :param self: the model instance
    :param dataset: the dataset
    :param val_set_split: the fraction of the replay buffer to be used as validation set
    """
    # if not need_aug:

    #     def refold_transform(x):
    #         return x.cpu()

    # else:
    #     data_shape = len(dataset.train_loader.dataset.data[0].shape)
    #     if data_shape == 3:

    #         def refold_transform(x):
    #             return (x.cpu() * 255).permute([0, 2, 3, 1]).numpy().astype(np.uint8)

    #     elif data_shape == 2:

    #         def refold_transform(x):
    #             return (x.cpu() * 255).squeeze(1).type(torch.uint8)

    if self.task > 0:
        buff_val_mask = torch.rand(len(self.buffer)) < val_set_split
        val_size = int(val_set_split * len(self.buffer))
        rand_indices = torch.randperm(len(dataset.train_loader.dataset))
        val_set = MammothSubset(dataset.train_loader.dataset, rand_indices[:val_size])
        train_set = MammothSubset(dataset.train_loader.dataset, rand_indices[val_size:])
        batch_size = dataset.train_loader.batch_size
        num_workers = dataset.train_loader.num_workers

        train_buffer_data = self.buffer.examples[: len(self.buffer)][~buff_val_mask]
        train_buffer_labels = self.buffer.labels[: len(self.buffer)][~buff_val_mask]
        train_buffer = MemoryDataset(
            train_buffer_data.cpu(),
            train_buffer_labels.cpu(),
        )

        dataset.train_loader = DataLoader(
            ConcatDataset([train_set]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        # Optionally split the replay buffer into a validation set
        if val_set_split > 0:
            raise NotImplementedError("Validation set not implemented yet for icarl")


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer(Dataset):
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, buffer_size, device, n_tasks=None, mode="reservoir"):
        assert mode in ("ring", "reservoir")
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == "ring":
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ["examples", "labels", "logits", "task_labels"]

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def init_tensors(
        self,
        examples: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
        task_labels: torch.Tensor,
    ) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith("els") else torch.float32
                setattr(
                    self,
                    attr_str,
                    torch.zeros(
                        (self.buffer_size, *attr.shape[1:]),
                        dtype=typ,
                        device=self.device,
                    ),
                )

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, "examples"):
            self.init_tensors(examples, labels, logits, task_labels)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)

    def get_data(
        self, size: int, transform: nn.Module = None, return_index=False
    ) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(
            min(self.num_seen_examples, self.examples.shape[0]),
            size=size,
            replace=False,
        )
        if transform is None:

            def transform(x):
                return x

        ret_tuple = (
            torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(
                self.device
            ),
        )
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(self.device),) + ret_tuple

    def get_data_by_index(self, indexes, transform: nn.Module = None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if transform is None:

            def transform(x):
                return x

        ret_tuple = (
            torch.stack([transform(ee.cpu()) for ee in self.examples[indexes]]).to(
                self.device
            ),
        )
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: nn.Module = None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None:

            def transform(x):
                return x

        ret_tuple = (
            torch.stack([transform(ee.cpu()) for ee in self.examples]).to(self.device),
        )
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
