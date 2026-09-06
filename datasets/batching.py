import random
import torch

class BatchIdxSampler_Class():
    def __init__(
        self,
        dataset=None,
        n=2,
        batch_size=16,
        sampler=None,
        drop_last=True,
        **kwargs,
    ):
        if dataset is None:
            dataset = sampler
        if not hasattr(dataset, "cls_datasets"):
            raise TypeError(
                "BatchIdxSampler_Class expects the original LMDB_Folder_Dataset. "
                "If this appears under DDP, set Trainer(replace_sampler_ddp=False). "
                f"Got {type(dataset).__name__}."
            )
        self.dataset = dataset
        self.sampler = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.n = n
        assert batch_size % n == 0, f"Number of videos per activity {n} does not work with this batch size {batch_size}"
        self.n_sampled_classes = int(batch_size / n)
        self.total_n_classes = len(self.dataset.cls_datasets)
        self._distributed_epoch = 0

    @staticmethod
    def _distributed_info():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def __len__(self):
        global_batches = len(self.dataset) // self.batch_size
        rank, world_size = self._distributed_info()
        return max((global_batches + world_size - 1 - rank) // world_size, 0)

    def __iter__(self):
        cls_idxs = []
        cls_lens = []
        for i in range(self.total_n_classes):
            if i == 0:
                cls_idxs.append(list(range(0, self.dataset.cls_end_idx[0])))
            else:
                cls_idxs.append(list(range(
                    self.dataset.cls_end_idx[i-1], self.dataset.cls_end_idx[i])))
            cls_lens.append(len(cls_idxs[-1]))
        nonempty_classes = [i for i in range(self.total_n_classes) if cls_lens[i] > 0]

        # import ipdb; ipdb.set_trace()
        # Lightning cannot inject DistributedSampler into this custom batch
        # sampler.  Generate the same global random stream on every rank and
        # let each rank consume a disjoint modulo shard.  Advancing the global
        # RNG for skipped batches is essential: otherwise every rank would
        # train on duplicate samples.
        rank, world_size = self._distributed_info()
        global_batches = len(self.dataset) // self.batch_size
        if world_size > 1:
            # Independent of process-global RNG state: all ranks reconstruct
            # the identical epoch stream before taking disjoint modulo shards.
            rng = random.Random(1729 + self._distributed_epoch)
            self._distributed_epoch += 1
        else:
            # Preserve the exact historical single-GPU sampling behavior.
            rng = random
        for global_batch_index in range(global_batches):
            batch_idxs = []
            chosen_classes = rng.sample(nonempty_classes, self.n_sampled_classes)
            for chosen_class in chosen_classes:
                chosen_class_idx = cls_idxs[chosen_class]
                if len(chosen_class_idx) < self.n:
                    chosen_class_idx = chosen_class_idx * (self.n // len(chosen_class_idx) + 1)
                sampled_cls_idxs = rng.sample(chosen_class_idx, self.n)
                batch_idxs.extend(sampled_cls_idxs)
            if global_batch_index % world_size == rank:
                yield batch_idxs


def flatten_batch(batch):
    keys = list(batch[0].keys())
    # from list of dicts to dict of lists
    packed_dict = {k: [dic[k] for dic in batch] for k in keys}
    for key, values in packed_dict.items():
        if key in ['cls', 'num_frames', 'num_steps']:
            packed_dict[key] = torch.stack(values, 0)
        if key in ['frame_features', 'step_features', 'step_ids', 'step_starts',
                   'step_ends', 'step_starts_sec', 'step_ends_sec']:
            packed_dict[key] = torch.cat(values, 0)
        if key in ['name', 'cls_name']:
            packed_dict[key] = '||'.join(values)
    return packed_dict


def unflatten_batch(flat_dict):
    unflat_dict = dict()

    # unflattening 0-dim features
    for key in ['cls', 'num_frames', 'num_steps']:
        unflat_dict[key] = torch.unbind(flat_dict[key], 0)

    # unflattening strings
    for key in ['name', 'cls_name']:
        unflat_dict[key] = flat_dict[key].split('||')

    # unflattening steps
    for key in ['step_features', 'step_ids', 'step_starts', 'step_ends', 'step_starts_sec', 'step_ends_sec']:
        unflat_dict[key] = []
        start = 0
        for num_steps in flat_dict['num_steps']:
            current_sample_steps = flat_dict[key][start:(start + num_steps)]
            unflat_dict[key].append(current_sample_steps)
            start += num_steps

    # unflattening frames
    for key in ['frame_features']:
        unflat_dict[key] = []
        start = 0
        for num_steps in flat_dict['num_frames']:
            current_sample_steps = flat_dict[key][start:(start + num_steps)]
            unflat_dict[key].append(current_sample_steps)
            start += num_steps

    # from dict of lists to list of dicts
    unflat_batch = [dict(zip(unflat_dict, t)) for t in zip(*unflat_dict.values())]
    return unflat_batch
