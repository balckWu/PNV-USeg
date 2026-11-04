"""Real six-domain 512x512 forward/backward and representative validation timing."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from tasksegment.configs import DOMAIN_NAMES
from tasksegment.data.datasets import set_seed
from tasksegment.training.trainer import _autocast, _make_grad_scaler
from tasksegment.training.metrics import binary_dice, binary_hd95, binary_iou
from campaign.model import CampaignModel, VARIANTS, state_hash
from campaign.train import ROOT, datasets, load_bank, manifest_for, episode, loss_terms, prototypes, json_write


def main():
    p = argparse.ArgumentParser(); p.add_argument('--variant', choices=VARIANTS, required=True)
    args = p.parse_args(); variant = args.variant
    gpu_map = json.loads((ROOT / 'campaign/gpu_map.json').read_text())
    assert VARIANTS[variant] != 3
    assert os.environ['CUDA_VISIBLE_DEVICES'] == gpu_map[str(VARIANTS[variant])]
    torch.set_num_threads(2); set_seed(42, True)
    assert torch.cuda.device_count() == 1
    device = torch.device('cuda:0')
    train, val = datasets(); manifest = manifest_for(train)
    set_seed(42, True)
    model = CampaignModel(load_bank()['features'], variant)
    initial_hash = state_hash(model)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scaler = _make_grad_scaler(True)
    times, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    for i, domain in enumerate(DOMAIN_NAMES):
        started = time.monotonic()
        xs, ys, xq, yq, _, _ = episode(train, domain, 0, i, device)
        with _autocast(device, True):
            output = model(xs, ys, xq, model.ids([domain] * 4, device))
            loss, _ = loss_terms(output, yq)
        assert torch.isfinite(loss)
        losses.append(float(loss.detach()))
        scaler.scale(loss / 4).backward()
        if i == 3:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            # Probe with a lower scale if FP16 warmup overflowed; normal training retains GradScaler.
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite unscaled gradient at real-size preflight')
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        del output, loss, xs, ys, xq, yq
        torch.cuda.synchronize()
        times.append(time.monotonic() - started)
        print('PREFLIGHT_TRAIN ' + json.dumps(dict(variant=variant, domain=domain, seconds=times[-1], loss=losses[-1])), flush=True)
    optimizer.zero_grad(set_to_none=True)
    bank = prototypes(model, train, manifest, device)
    val_times = {}
    with torch.inference_mode():
        for domain in DOMAIN_NAMES:
            selected = val.organ_to_indices[domain][:4]
            started = time.monotonic()
            xq = torch.stack([val[i][0] for i in selected]).to(device)
            output = model.segment(xq, bank[domain].to(device), model.ids([domain] * len(selected), device))
            prob = output['pred_masks'].softmax(1)[:, 1]
            assert torch.isfinite(prob).all()
            pred = (prob >= .5).cpu().numpy().astype(np.uint8)
            for j, index in enumerate(selected):
                target = val.load_resized_mask(index)
                binary_dice(pred[j], target); binary_iou(pred[j], target); binary_hd95(pred[j], target)
            torch.cuda.synchronize()
            val_times[domain] = dict(seconds=time.monotonic() - started, cases=len(selected))
            print('PREFLIGHT_VALIDATION ' + json.dumps(dict(variant=variant, domain=domain, **val_times[domain])), flush=True)
    record = dict(passed=True, variant=variant, physical_gpu=VARIANTS[variant],
                  initial_state_sha256=initial_hash, manifest=manifest,
                  real_size=[512, 512], support_batch=4, query_batch=4,
                  train_step_seconds=times, losses=losses, validation_sample_timings=val_times,
                  peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
                  device_total_GiB=torch.cuda.get_device_properties(0).total_memory / 2**30)
    assert record['peak_reserved_GiB'] < record['device_total_GiB'] - .3
    json_write(ROOT / f'records/preflight_{variant}.json', record)
    print('PREFLIGHT_PASSED ' + json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
