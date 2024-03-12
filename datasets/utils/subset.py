from torch.utils.data import Subset


class MammothSubset(Subset):
    def __getitem__(self, idx):
        if hasattr(self, "logits"):
            return *super().__getitem__(idx), self.logits[idx]
        return super().__getitem__(idx)
