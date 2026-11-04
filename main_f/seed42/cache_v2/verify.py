"""GPU equivalence checks and alternating old/new throughput measurements."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch

from campaign import train as ref
from cache_v2.input_cache import InputCache
from cache_v2 import engine


def digest(value):
    h = hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            h.update(str((str(x.dtype), tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                h.update(str(key).encode()); add(x[key])
        elif isinstance(x, (list, tuple)):
            for item in x: add(item)
        else:
            h.update(repr(x).encode())
    add(value)
    return h.hexdigest()


def cpu_snapshot(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(v) for v in value)
    return value


def difference(a, b):
    maximum, square_error, square_reference = 0., 0., 0.
    def compare(x, y):
        nonlocal maximum, square_error, square_reference
        if torch.is_tensor(x):
            if x.shape != y.shape or x.dtype != y.dtype:
                raise AssertionError('Tensor structure changed')
            if not x.is_floating_point():
                if not torch.equal(x, y): raise AssertionError('Integer state changed')
                return
            finite = torch.isfinite(x) & torch.isfinite(y)
            if not torch.equal(torch.isnan(x), torch.isnan(y)) or not torch.equal(torch.isinf(x), torch.isinf(y)):
                raise AssertionError('Nonfinite state differs')
            xx, yy = x[finite], y[finite]
            delta = (xx - yy).float()
            if delta.numel():
                maximum = max(maximum, float(delta.abs().max()))
                square_error += float(delta.square().sum(dtype=torch.float64))
                square_reference += float(xx.float().square().sum(dtype=torch.float64))
        elif isinstance(x, dict):
            if x.keys() != y.keys(): raise AssertionError('State keys changed')
            for k in x: compare(x[k], y[k])
        elif isinstance(x, (list, tuple)):
            if len(x) != len(y): raise AssertionError('State length changed')
            for u, v in zip(x, y): compare(u, v)
        elif x != y:
            raise AssertionError('Nontensor state changed')
    compare(a, b)
    return dict(max_abs=maximum, relative_l2=(square_error / max(square_reference, 1e-30)) ** .5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True, choices=ref.VARIANTS)
    args = p.parse_args()
    variant = args.variant
    mapping = json.loads((ref.ROOT / 'campaign/gpu_map.json').read_text())
    uuid = mapping[str(ref.VARIANTS[variant])]
    if ref.VARIANTS[variant] == 3 or uuid.endswith('6468') or os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise RuntimeError('GPU UUID guard failed')
    torch.set_num_threads(2)
    ref.set_seed(42, deterministic=True)
    device = torch.device('cuda:0')
    ck = torch.load(ref.ROOT / 'archives/cache_v2_20261026' / variant / 'last.pt', map_location='cpu', weights_only=True)
    train, val = ref.datasets()
    bank = ref.load_bank()
    model = ref.CampaignModel(bank['features'], variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scaler = ref._make_grad_scaler(True)
    cache = InputCache(train, val, device)
    epoch, first = ck['epoch_index'], ck['next_step']
    result = dict(variant=variant, physical_gpu=ref.VARIANTS[variant], uuid=uuid,
                  epoch=epoch + 1, next_step=first, checkpoint_model_sha256=digest(ck['model']),
                  checkpoint_optimizer_sha256=digest(ck['optimizer']),
                  cache_build_seconds=cache.build_seconds, cache_gpu_GiB=cache.gpu_bytes / 2**30)
    input_checks = []
    for i, domain in enumerate(ref.DOMAIN_NAMES):
        for offset in (0, 37):
            step = (first + i * 19 + offset) % 1500
            old = ref.episode(train, domain, epoch, step, device)
            new = cache.episode(domain, epoch, step)
            if old[4:] != new[4:] or not all(torch.equal(a, b) for a, b in zip(old[:4], new[:4])):
                raise AssertionError('Input/augmentation differs for ' + domain)
            input_checks.append(dict(domain=domain, step=step, bitwise_equal=True))
    result['input_checks'] = input_checks
    schedule = ref.domain_schedule(epoch)
    def update(old):
        engine.restore(model, optimizer, scaler, ck)
        model.train()
        losses = []
        accum = torch.zeros(5, device=device, dtype=torch.float64)
        for step in range(first, first + 4):
            loss, terms, _, _ = engine.one_episode(model, cache, schedule[step], epoch, step, device, old=old)
            if old:
                if not torch.isfinite(loss): raise FloatingPointError()
            else:
                torch._assert_async(torch.isfinite(loss), 'Nonfinite check')
            scaler.scale(loss / 4.).backward()
            values = torch.stack([loss.detach()] + [terms[k].detach() for k in ('main', 'aux', 'dense', 'semantic')]).double()
            accum.add_(values)
            losses.append(float(loss.detach()))
            del loss, terms
        scaler.unscale_(optimizer)
        grad_state = cpu_snapshot({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
        gradients = digest(grad_state)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        snapshot = dict(gradients=grad_state, model=cpu_snapshot(model.state_dict()),
                        optimizer=cpu_snapshot(optimizer.state_dict()))
        return dict(losses=losses, sums=accum.cpu().tolist(), gradients=gradients,
                    gradient_norm=float(norm), model=digest(model.state_dict()),
                    optimizer=digest(optimizer.state_dict()), scaler=digest(scaler.state_dict()),
                    rng=digest([torch.get_rng_state(), torch.cuda.get_rng_state(0), random.getstate()])), snapshot
    old_update, old_state = update(True)
    repeated_update, repeated_state = update(True)
    baseline_diff = {k: difference(old_state[k], repeated_state[k]) for k in old_state}
    del repeated_state
    new_update, new_state = update(False)
    cached_diff = {k: difference(old_state[k], new_state[k]) for k in old_state}
    del old_state, new_state
    result['reference_update'] = old_update
    result['cached_update'] = new_update
    result['original_repeat_update'] = repeated_update
    result['original_repeat_difference'] = baseline_diff
    result['cached_difference'] = cached_diff
    result['optimizer_update_bitwise_equal'] = old_update == new_update
    result['original_repeat_bitwise_equal'] = old_update == repeated_update
    for key in ('rng', 'scaler'):
        if old_update[key] != new_update[key] or old_update[key] != repeated_update[key]:
            raise AssertionError('Random/scaler state changed: ' + key)
    floors = {'gradients': (1e-6, 1e-5), 'model': (1e-7, 1e-7), 'optimizer': (1e-6, 1e-5)}
    for key, (absolute, relative) in floors.items():
        for metric, floor in (('max_abs', absolute), ('relative_l2', relative)):
            if cached_diff[key][metric] > max(5 * baseline_diff[key][metric], floor):
                ref.json_write(ref.ROOT / 'records' / ('cache_v2_difference_failed_' + variant + '.json'), result)
                raise AssertionError('Difference exceeds measured original variation: ' + key + '/' + metric)
    if not np.allclose(old_update['losses'], new_update['losses'], rtol=1e-5, atol=1e-6):
        raise AssertionError('Losses changed materially')
    result['update_within_reference_numerical_variation'] = True
    print('UPDATE_EQUIVALENCE ' + json.dumps(dict(variant=variant, original_repeat=baseline_diff, cached=cached_diff)), flush=True)
    # Verify representative FP32 evaluation predictions and support tokens in
    # all six domains, using the same unchanged checkpoint for both paths.
    engine.restore(model, optimizer, scaler, ck)
    model.eval()
    manifest = ck['config']['support_manifest']
    with torch.inference_mode():
        old_bank = ref.prototypes(model, train, manifest, device)
        new_bank = engine.prototypes(model, cache, manifest, device)
        cache.stage_validation()
        evaluation_checks = []
        for domain in ref.DOMAIN_NAMES:
            if not torch.equal(old_bank[domain].to(device), new_bank[domain]):
                raise AssertionError('Support bank differs')
            indices = val.organ_to_indices[domain][:4]
            a = torch.stack([val[i][0] for i in indices]).to(device)
            b = cache.validation_batch(indices)
            if not torch.equal(a, b): raise AssertionError('Validation input differs')
            ids = model.ids([domain] * len(indices), device)
            logits_a = model.segment(a, old_bank[domain].to(device), ids)['pred_masks']
            logits_b = model.segment(b, new_bank[domain], ids)['pred_masks']
            if not torch.equal(logits_a, logits_b): raise AssertionError('Validation logits differ')
            for i in indices:
                if not np.array_equal(val.load_resized_mask(i), cache.val_masks[i]):
                    raise AssertionError('Validation target differs')
            evaluation_checks.append(dict(domain=domain, cases=len(indices), logits_bitwise_equal=True))
        cache.release_validation()
    result['evaluation_checks'] = evaluation_checks
    def bench(old):
        engine.restore(model, optimizer, scaler, ck)
        model.train()
        values = torch.zeros(5, device=device, dtype=torch.float64)
        torch.cuda.synchronize()
        begin = time.monotonic(); measured = None
        for offset in range(32):
            if offset == 8:
                torch.cuda.synchronize(); measured = time.monotonic()
            step = first + offset
            domain = schedule[step % 1500]
            loss, terms, _, _ = engine.one_episode(model, cache, domain, epoch, step, device, old=old)
            if old:
                if not torch.isfinite(loss): raise FloatingPointError()
            else:
                torch._assert_async(torch.isfinite(loss), 'Nonfinite benchmark loss')
            scaler.scale(loss / 4.).backward()
            if old:
                float(loss.detach())
                for term in terms.values(): float(term.detach())
            else:
                values.add_(torch.stack([loss.detach()] + [terms[k].detach() for k in ('main', 'aux', 'dense', 'semantic')]).double())
            del loss, terms
            if (offset + 1) % 4 == 0:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if (offset + 1) % 20 == 0: torch.cuda.synchronize()
        torch.cuda.synchronize()
        return (time.monotonic() - measured) / 24
    timings = []
    for old in (True, False, False, True):
        seconds = bench(old)
        timings.append(dict(mode='original' if old else 'cached', seconds_per_episode=seconds))
        print('BENCH ' + json.dumps(dict(variant=variant, **timings[-1])), flush=True)
    a = statistics.mean(x['seconds_per_episode'] for x in timings if x['mode'] == 'original')
    b = statistics.mean(x['seconds_per_episode'] for x in timings if x['mode'] == 'cached')
    if digest(ck['optimizer']) != result['checkpoint_optimizer_sha256']:
        raise AssertionError('Diagnostic restore mutated the input checkpoint')
    if variant == 'F':
        engine.restore(model, optimizer, scaler, ck)
        torch.cuda.synchronize(); began = time.monotonic()
        old_bank = ref.prototypes(model, train, manifest, device)
        original_metrics = ref.evaluate(model, val, old_bank, device)
        original_seconds = time.monotonic() - began
        torch.cuda.synchronize(); began = time.monotonic()
        new_bank = engine.prototypes(model, cache, manifest, device)
        cached_metrics = engine.evaluate(model, cache, new_bank, device)
        cached_seconds = time.monotonic() - began
        full = dict(variant=variant, original=original_metrics, cached=cached_metrics,
                    exact_equal=original_metrics == cached_metrics,
                    original_seconds=original_seconds, cached_seconds=cached_seconds, cases=1361)
        ref.json_write(ref.ROOT / 'records/cache_v2_full_validation.json', full)
        if not full['exact_equal']:
            raise AssertionError('Full validation metrics changed')
        print('FULL_VALIDATION_PASSED ' + json.dumps({k: full[k] for k in ('exact_equal', 'original_seconds', 'cached_seconds', 'cases')}), flush=True)
    result.update(benchmark=timings, original_seconds_per_episode=a, cached_seconds_per_episode=b,
                  speedup=a / b, training_time_reduction=1 - b / a,
                  peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                  reserved_GiB=torch.cuda.memory_reserved() / 2**30, passed=True)
    ref.json_write(ref.ROOT / 'records' / ('cache_v2_verify_' + variant + '.json'), result)
    print('VERIFY_PASSED ' + json.dumps({k: result[k] for k in ('variant', 'speedup', 'training_time_reduction', 'peak_allocated_GiB', 'reserved_GiB')}), flush=True)


if __name__ == '__main__':
    main()
