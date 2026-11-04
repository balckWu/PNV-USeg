import time
import numpy as np
import torch

from campaign import train as ref


class InputCache:
    """Exact CPU-normalized float32 inputs on GPU; bit-packed binary labels.

    The validation images stay in pinned host memory during training and are
    staged to GPU only during validation. No encoder outputs are cached.
    """
    def __init__(self, train, val, device):
        self.train, self.val, self.device = train, val, device
        self.shape = (512, 512)
        self.bit_shifts = torch.arange(8, device=device, dtype=torch.uint8)
        started = time.monotonic()
        self.images = torch.empty((len(train), 1, 512, 512), dtype=torch.float32, device=device)
        self.masks = torch.empty((len(train), 512 * 512 // 8), dtype=torch.uint8, device=device)
        # Chunk uploads avoid thousands of tiny synchronous copies. Normalization
        # is performed by the original CPU function, preserving every input bit.
        for first in range(0, len(train), 32):
            selected = list(range(first, min(first + 32, len(train))))
            pairs = [train[i] for i in selected]
            images = torch.stack([p[0] for p in pairs])
            packed = np.stack([np.packbits(p[1].numpy().astype(np.uint8).reshape(-1), bitorder='little') for p in pairs])
            self.images[first:first + len(selected)].copy_(images)
            self.masks[first:first + len(selected)].copy_(torch.from_numpy(packed))
        self.val_images_host = torch.empty((len(val), 1, 512, 512), dtype=torch.float32, pin_memory=True)
        self.val_masks = np.empty((len(val), 512, 512), dtype=np.uint8)
        for i in range(len(val)):
            image, mask, _ = val[i]
            self.val_images_host[i].copy_(image)
            self.val_masks[i] = mask.numpy().astype(np.uint8)
        self.val_images = None
        torch.cuda.synchronize(device)
        self.build_seconds = time.monotonic() - started
        self.gpu_bytes = self.images.numel() * self.images.element_size() + self.masks.numel()
        print('INPUT_CACHE_READY ' + str(dict(seconds=self.build_seconds, gpu_GiB=self.gpu_bytes / 2**30,
                                             train=len(train), validation=len(val))), flush=True)

    def training_batch(self, indices):
        ids = torch.tensor(indices, dtype=torch.long, device=self.device)
        image = self.images.index_select(0, ids)
        packed = self.masks.index_select(0, ids)
        mask = ((packed[:, :, None] >> self.bit_shifts) & 1).reshape(-1, 1, 512, 512).float()
        return image, mask

    def episode(self, domain, epoch, step):
        seed = ref.SEED + epoch * 1000003 + step * 1009
        with ref.data_rng(seed):
            support, query = self.train.sample_episode_indices(domain, 4, 4)
            if set(support) & set(query):
                raise ValueError('Support/query overlap')
            images, masks = self.training_batch(support + query)
            xs, xq = images[:4], images[4:]
            ys, yq = masks[:4], masks[4:]
            modality = ref.MODALITY.get(domain, 'generic')
            xs, ys = ref.augment_episode_medical(xs, ys, modality)
            xq, yq = ref.augment_episode_medical(xq, yq, modality)
        return xs, ys, xq, yq.squeeze(1).long(), support, query

    def stage_validation(self):
        if self.val_images is not None:
            raise RuntimeError('Validation cache is already staged')
        self.val_images = self.val_images_host.to(self.device, non_blocking=True)

    def validation_batch(self, indices):
        ids = torch.tensor(indices, dtype=torch.long, device=self.device)
        return self.val_images.index_select(0, ids)

    def release_validation(self):
        self.val_images = None

    def discard_redundant_raw_cache(self):
        self.train.clear_cache()
        self.val.clear_cache()
