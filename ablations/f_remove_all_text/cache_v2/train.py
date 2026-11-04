"""Resume the original experiment with exact input caches and less host waiting."""
import argparse
import hashlib
import json
import os
import signal
import time

import numpy as np
import torch

from campaign import train as ref
from cache_v2.input_cache import InputCache
from cache_v2 import engine

ROOT = ref.ROOT


def verify_revision():
    for filename in ('bundle_manifest.json', 'cache_v2/manifest.json'):
        manifest = json.loads((ROOT / filename).read_text())
        for name, expected in manifest['files'].items():
            if ref.file_hash(ROOT / name) != expected:
                raise RuntimeError('Source fingerprint mismatch: ' + name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True, choices=ref.VARIANTS)
    args = parser.parse_args()
    variant = args.variant
    expected = json.loads((ROOT / 'campaign/gpu_map.json').read_text())[str(ref.VARIANTS[variant])]
    if ref.VARIANTS[variant] == 3 or expected.endswith('6468') or os.environ.get('CUDA_VISIBLE_DEVICES') != expected:
        raise RuntimeError('GPU guard rejected resume')
    verify_revision()
    torch.set_num_threads(2)
    ref.set_seed(42, deterministic=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one explicitly allowed GPU required')
    device = torch.device('cuda:0')
    out = ROOT / 'runs' / variant / 'seed_42'
    original = json.loads((out / 'run.json').read_text())
    checkpoint = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
    for name in ('variant', 'source_sha256', 'text_bank_sha256', 'train_data_sha256', 'val_data_sha256', 'support_manifest'):
        if checkpoint['config'][name] != original[name]:
            raise RuntimeError('Original checkpoint/run mismatch: ' + name)
    protocol = json.loads((ROOT / 'campaign/protocol.json').read_text())
    if original['protocol'] != protocol or checkpoint['config']['protocol'] != protocol:
        raise RuntimeError('Scientific protocol changed')
    if original['source_sha256'] != ref.file_hash(ROOT / 'bundle_manifest.json'):
        raise RuntimeError('Original source manifest changed')
    train, val = ref.datasets()
    manifest = ref.manifest_for(train)
    if manifest != original['support_manifest']:
        raise RuntimeError('Support manifest changed')
    if ref.data_fingerprint(train) != original['train_data_sha256'] or ref.data_fingerprint(val) != original['val_data_sha256']:
        raise RuntimeError('Data changed')
    bank = ref.load_bank()
    if ref.file_hash(ROOT / 'campaign/text_bank.pt') != original['text_bank_sha256']:
        raise RuntimeError('Text changed')
    ref.set_seed(42, deterministic=True)
    model = ref.CampaignModel(bank['features'], variant)
    if ref.state_hash(model) != original['initial_state_sha256']:
        raise RuntimeError('Common initialization changed')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scaler = ref._make_grad_scaler(True)
    cache = InputCache(train, val, device)
    cache.discard_redundant_raw_cache()
    cfg = dict(original)
    cfg['execution_revision'] = dict(name='cache_v2_20261026',
        manifest_sha256=ref.file_hash(ROOT / 'cache_v2/manifest.json'),
        checkpoint_parent_sha256=ref.file_hash(out / 'last.pt'),
        original_archive='archives/cache_v2_20261026/' + variant,
        input_cache_gpu_GiB=cache.gpu_bytes / 2**30,
        preserves_scientific_protocol=True)
    best, bad = checkpoint['best_val_dice'], checkpoint['bad_epochs']
    start_epoch, start_step, old_sums = checkpoint['epoch_index'], checkpoint['next_step'], checkpoint['sums']
    engine.restore(model, optimizer, scaler, checkpoint)
    resumed_hash = ref.state_hash(model)
    ref.json_write(out / 'run_cache_v2.json', dict(config=cfg, resumed_epoch=start_epoch + 1,
        resumed_next_step=start_step, resumed_model_sha256=resumed_hash,
        optimizer_state_count=len(optimizer.state), scaler=scaler.state_dict(),
        best_val_dice=best, bad_epochs=bad, cache_build_seconds=cache.build_seconds))
    del checkpoint
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, ref.request_stop)
    stopped = out / 'stopped.json'
    if stopped.exists():
        stopped.rename(out / 'stopped_before_cache_v2.json')
    print('CACHE_RESUME_READY ' + json.dumps(dict(variant=variant, epoch=start_epoch + 1,
          next_step=start_step, model_sha256=resumed_hash, execution_revision=cfg['execution_revision'])), flush=True)
    for epoch in range(start_epoch, 150):
        lr = ref._set_cosine_lr(optimizer, epoch, 150, 5e-5, 1e-6, 5)
        names = ('loss', 'main', 'aux', 'dense', 'semantic')
        sums = old_sums if epoch == start_epoch and start_step else {k: 0. for k in names}
        sums_gpu = torch.tensor([sums[k] for k in names], dtype=torch.float64, device=device)
        schedule = ref.domain_schedule(epoch)
        first = start_step if epoch == start_epoch else 0
        sample_hash = hashlib.sha256()
        if first:
            # The old checkpoint did not store a partially built SHA256 object.
            # Reconstruct the deterministic sampling prefix without augmenting.
            numpy_state = np.random.get_state()
            for prior in range(first):
                np.random.seed((42 + epoch * 1000003 + prior * 1009) % 2**32)
                s, q = train.sample_episode_indices(schedule[prior], 4, 4)
                sample_hash.update(json.dumps([schedule[prior], s, q]).encode())
            np.random.set_state(numpy_state)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.monotonic()
        for step in range(first, 1500):
            loss, terms, support, query = engine.one_episode(model, cache, schedule[step], epoch, step, device)
            sample_hash.update(json.dumps([schedule[step], support, query]).encode())
            torch._assert_async(torch.isfinite(loss), 'Nonfinite loss in cached training')
            scaler.scale(loss / 4.).backward()
            current_loss = loss.detach()
            sums_gpu.add_(torch.stack([current_loss] + [terms[k].detach() for k in names[1:]]).double())
            del loss, terms
            update_due = (step + 1) % 4 == 0
            if update_due:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 20 == 0 or step == first:
                torch.cuda.synchronize()
                now = time.monotonic()
                ref.json_write(out / 'status.json', dict(variant=variant, epoch=epoch + 1,
                    step=step + 1, steps_per_epoch=1500, loss=float(current_loss),
                    elapsed_epoch_seconds=now - epoch_start, lr=lr,
                    allocated_GiB=torch.cuda.memory_allocated() / 2**30,
                    peak_GiB=torch.cuda.max_memory_allocated() / 2**30,
                    scaler=float(scaler.get_scale()), execution_revision='cache_v2_20261026',
                    resumed_epoch_first_step=first))
                print('STEP ' + (out / 'status.json').read_text().replace('\n', ''), flush=True)
            if (step + 1) % 300 == 0 or (ref.STOP and update_due):
                sums = dict(zip(names, sums_gpu.cpu().tolist()))
                ref.save_checkpoint(out / 'last.pt', model, optimizer, scaler, cfg, epoch, step + 1, best, bad, sums)
            if ref.STOP and update_due:
                ref.json_write(out / 'stopped.json', dict(epoch=epoch + 1, next_step=step + 1))
                return
        torch.cuda.synchronize()
        train_seconds = time.monotonic() - epoch_start
        sums = dict(zip(names, sums_gpu.cpu().tolist()))
        val_start = time.monotonic()
        visual_bank = engine.prototypes(model, cache, manifest, device)
        metrics = engine.evaluate(model, cache, visual_bank, device)
        val_seconds = time.monotonic() - val_start
        improved = metrics['macro_dice'] > best
        if improved:
            best, bad = metrics['macro_dice'], 0
        else:
            bad += 1
        row = dict(epoch=epoch + 1, **metrics, **{'train_' + k: sums[k] / 1500 for k in names},
            lr=lr, train_seconds=train_seconds, validation_seconds=val_seconds,
            train_seconds_is_partial=bool(first), measured_training_episodes=1500 - first,
            sample_sequence_sha256=sample_hash.hexdigest(), best_val_dice=best, bad_epochs=bad,
            execution_revision='cache_v2_20261026')
        ref.json_write(out / ('validation_epoch_%03d.json' % (epoch + 1)), row)
        with (out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        if improved:
            ref.save_checkpoint(out / 'best.pt', model, optimizer, scaler, cfg, epoch + 1, 0, best, bad, {}, visual_bank)
        ref.save_checkpoint(out / 'last.pt', model, optimizer, scaler, cfg, epoch + 1, 0, best, bad, {}, visual_bank)
        print('EPOCH ' + json.dumps(row), flush=True)
        start_step = 0
        if bad >= 20 or ref.STOP:
            break
    ref.json_write(out / ('stopped.json' if ref.STOP else 'complete.json'),
        dict(best_val_dice=best, last_epoch=epoch + 1, early_stopped=bad >= 20, test_evaluated=False))


if __name__ == '__main__':
    main()
