import numpy as np
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler[list]):
    """Yields batches with configurable positive/negative ratio.

    Default: ~50% positive (BGC) and ~50% negative windows (legacy behavior).
    Set pos_ratio < 0.5 for deployment-realistic priors (e.g. 0.15 for ~15% BGC).

    Within positives, we try to give roughly even coverage of the 7 types
    by sampling types in round-robin.
    """

    def __init__(self, labels: np.ndarray, batch_size: int = 128,
                 seed: int = 0, drop_last: bool = True,
                 pos_ratio: float = 0.5,
                 hard_neg_indices: np.ndarray | None = None,
                 hard_neg_ratio: float = 0.0):
        # labels: (N, 8); col 7 == is_negative
        self.labels = labels
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.pos_ratio = pos_ratio

        is_neg = labels[:, 7].astype(bool)
        self.pos_idx = np.where(~is_neg)[0]
        self.neg_idx = np.where(is_neg)[0]

        self.n_pos = max(1, int(batch_size * pos_ratio))
        self.n_neg = batch_size - self.n_pos

        # Hard-negative upsampling: a fraction of the negative half
        # comes from the hard-negative pool instead of uniform sampling.
        self.hard_neg_ratio = hard_neg_ratio
        if hard_neg_indices is not None and len(hard_neg_indices) > 0 and hard_neg_ratio > 0:
            self.hard_neg_idx = hard_neg_indices
            self.n_hard = max(1, int(self.n_neg * hard_neg_ratio))
            self.n_easy = self.n_neg - self.n_hard
        else:
            self.hard_neg_idx = None
            self.n_hard = 0
            self.n_easy = self.n_neg

        # Per-type positive index pools
        self.n_types = labels.shape[1] - 1
        self.type_pools = [
            np.where((labels[:, c] == 1) & (~is_neg))[0]
            for c in range(self.n_types)
        ]

        # steps per epoch: bounded by smaller class
        steps_from_pos = len(self.pos_idx) // self.n_pos
        steps_from_neg = len(self.neg_idx) // self.n_neg
        self._len = max(1, min(steps_from_pos, steps_from_neg))

        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)

        pos_perm = rng.permutation(self.pos_idx)
        neg_perm = rng.permutation(self.neg_idx)
        hard_perm = rng.permutation(self.hard_neg_idx) if self.hard_neg_idx is not None else None
        h_cur = 0

        # Round-robin type pools (shuffled within each type)
        type_perms = [rng.permutation(p) for p in self.type_pools]
        type_cursors = [0] * self.n_types

        p_cur, n_cur = 0, 0
        for _ in range(self._len):
            # Fill positive half with round-robin across types (falls back to
            # raw pos permutation when a type pool is exhausted).
            pos_batch = []
            tc = 0
            while len(pos_batch) < self.n_pos:
                pool = type_perms[tc % self.n_types]
                cur = type_cursors[tc % self.n_types]
                if len(pool) == 0:
                    # this type has no positives — skip
                    if all(len(p) == 0 for p in type_perms):
                        break
                    tc += 1
                    continue
                idx = pool[cur % len(pool)]
                type_cursors[tc % self.n_types] = (cur + 1)
                pos_batch.append(int(idx))
                tc += 1

            # top-up if still short (e.g. only 1 type had samples)
            while len(pos_batch) < self.n_pos:
                pos_batch.append(int(pos_perm[p_cur % len(pos_perm)]))
                p_cur += 1

            # Negative half: mix of hard negatives and easy negatives
            neg_batch = []
            if self.hard_neg_idx is not None and self.n_hard > 0:
                for _k in range(self.n_hard):
                    neg_batch.append(int(hard_perm[h_cur % len(hard_perm)]))
                    h_cur += 1
                for _k in range(self.n_easy):
                    neg_batch.append(int(neg_perm[n_cur % len(neg_perm)]))
                    n_cur += 1
            else:
                for _k in range(self.n_neg):
                    neg_batch.append(int(neg_perm[n_cur % len(neg_perm)]))
                    n_cur += 1

            batch = pos_batch + neg_batch
            rng.shuffle(batch)
            yield batch
