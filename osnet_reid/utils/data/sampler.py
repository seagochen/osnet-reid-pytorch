"""
Random identity sampler for ReID training.

Constructs batches with P identities and K images per identity,
which is required for triplet loss with batch hard mining.
"""
import copy
import random
from collections import defaultdict
from torch.utils.data import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly samples P identities, then for each identity,
    randomly samples K instances, forming a batch of P * K.

    Args:
        dataset: ReIDDataset with pid_index attribute
        num_instances: Number of images per identity in a batch (K)
    """

    def __init__(self, dataset, num_instances=4, batch_size=64):
        self.pid_index = dataset.pid_index
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances

        # Flatten: create index list
        self.pids = list(self.pid_index.keys())
        self.length = 0
        for pid in self.pids:
            num = len(self.pid_index[pid])
            num = max(num, self.num_instances)
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.pid_index[pid])
            if len(idxs) < self.num_instances:
                idxs = idxs * (self.num_instances // len(idxs) + 1)
            random.shuffle(idxs)

            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avail_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avail_pids) >= 2:
            selected_pids = random.sample(avail_pids, min(len(avail_pids), self.num_pids_per_batch))
            for pid in selected_pids:
                if batch_idxs_dict[pid]:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    final_idxs.extend(batch_idxs)
                else:
                    avail_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length
