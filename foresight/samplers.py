"""Validation partitions without duplicated evaluated samples."""

from torch.utils.data import DistributedSampler


class ValidationSampler(DistributedSampler):
    """Give an empty rank a placeholder so every rank reaches epoch-end collectives."""

    def __iter__(self):
        indices = range(self.rank, len(self.dataset), self.num_replicas)
        return iter(indices if len(indices) else [0])

    def __len__(self):
        return max(1, len(range(self.rank, len(self.dataset), self.num_replicas)))
