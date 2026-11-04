"""Seed3407/2026 of F, from scratch, using the previously verified exact cache path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(3407, 2026), required=True)
    parser.add_argument('--physical-gpu', type=int, choices=(0, 1, 2, 4, 5, 6), required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    from paperlab.gpu import gpu_environment
    expected = gpu_environment(args.physical_gpu)['CUDA_VISIBLE_DEVICES']
    mapping = json.loads((ROOT / 'campaign/gpu_map.json').read_text())
    if expected != mapping[str(args.physical_gpu)] or os.environ.get('CUDA_VISIBLE_DEVICES') != expected:
        raise RuntimeError('Live physical GPU/UUID and inherited environment differ')
    if os.environ.get('USTS_TRAIN_SEED') != str(args.seed):
        raise RuntimeError('Explicit training seed environment differs from CLI seed')
    for name, sha in json.loads((ROOT / 'bundle_manifest.json').read_text())['files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise RuntimeError('Isolated source fingerprint mismatch: ' + name)
    import numpy as np
    import torch
    from campaign import train as ref
    from cache_v2.input_cache import InputCache
    from cache_v2 import engine
    if ref.SEED != args.seed:
        raise RuntimeError('Sampling/augmentation/support seed adapter failed')
    torch.set_num_threads(2)
    ref.set_seed(args.seed, deterministic=True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one explicitly allowed GPU required')
    device = torch.device('cuda:0')
    variant = 'F'
    seed = args.seed
    out = ROOT / 'runs/F' / ('seed_' + str(seed))
    if not args.preflight and (out / 'run.json').exists():
        raise FileExistsError('Refusing to overwrite or implicitly resume an existing run')
    train, val = ref.datasets()
    manifest = ref.manifest_for(train)
    bank = ref.load_bank()
    ref.set_seed(args.seed, deterministic=True)
    model = ref.CampaignModel(bank['features'], variant)
    from structural_init import initialize
    initialization_audit=initialize(model,bank['features'])
    initial_hash = ref.state_hash(model)
    protocol = json.loads((ROOT / 'campaign/protocol.json').read_text())
    protocol['training_seed'] = args.seed
    protocol['condition_ablation'] = 'remove_summary'
    cfg = dict(protocol=protocol, variant=variant, seed=seed, physical_gpu=args.physical_gpu,
        gpu_uuid=expected, source_sha256=ref.file_hash(ROOT / 'bundle_manifest.json'),
        parent_source_manifest_sha256=ref.file_hash(ROOT / 'parent_bundle_manifest.json'),
        text_bank_sha256=ref.file_hash(ROOT / 'campaign/text_bank.pt'),
        train_data_sha256=ref.data_fingerprint(train), val_data_sha256=ref.data_fingerprint(val),
        support_manifest=manifest, initial_state_sha256=initial_hash, from_scratch=True,initialization_audit=initialization_audit,
        parameters=sum(p.numel() for p in model.parameters()),
        execution_revision=dict(name='fresh_cached_f_multiseed_20261027', preserves_scientific_protocol=True))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scaler = ref._make_grad_scaler(True)
    if args.preflight:scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    cache = InputCache(train, val, device)
    cache.discard_redundant_raw_cache()
    if args.preflight:
        for index, domain in enumerate(ref.DOMAIN_NAMES):
            old = ref.episode(train, domain, 0, index, device)
            new = cache.episode(domain, 0, index)
            if old[4:] != new[4:] or not all(torch.equal(a, b) for a, b in zip(old[:4], new[:4])):
                raise RuntimeError('Seeded cached sampling/augmentation input changed: ' + domain)
        model.train()
        schedule = list(ref.DOMAIN_NAMES)
        first_lr = ref._set_cosine_lr(optimizer, 0, 150, 5e-5, 1e-6, 5)
        losses = []
        for step in range(6):
            loss, terms, _, _ = engine.one_episode(model, cache, schedule[step], 0, step, device)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite preflight loss')
            scaler.scale(loss / 4.).backward()
            losses.append(float(loss.detach()))
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if not torch.isfinite(norm): raise RuntimeError('Nonfinite gradient norm')
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        if not optimizer.state: raise RuntimeError('AdamW did not update any parameters')
        model.eval()
        with torch.inference_mode():
            visual = engine.prototypes(model, cache, manifest, device)
            samples = val.organ_to_indices[ref.DOMAIN_NAMES[0]][:4]
            images = torch.stack([val[i][0] for i in samples]).to(device)
            result = model.segment(images, visual[ref.DOMAIN_NAMES[0]], model.ids([ref.DOMAIN_NAMES[0]] * len(samples), device))
            if not torch.isfinite(result['pred_masks']).all(): raise RuntimeError('Nonfinite validation output')
        cfg.update(passed=True, preflight_episodes=6, real_adamw_update=True, all_six_domain_inputs_bitwise_equal=True,
                   losses=losses, first_lr=first_lr, peak_GiB=torch.cuda.max_memory_allocated()/2**30)
        ref.json_write(ROOT / 'records' / ('preflight_seed' + str(seed) + '.json'), cfg)
        print('PREFLIGHT_PASSED ' + json.dumps(cfg), flush=True)
        return
    preflight = json.loads((ROOT / 'records' / ('preflight_seed' + str(seed) + '.json')).read_text())
    for name in ('initial_state_sha256', 'support_manifest', 'train_data_sha256', 'val_data_sha256', 'source_sha256'):
        if preflight[name] != cfg[name]: raise RuntimeError('Fresh restart differs from preflight: ' + name)
    ref.json_write(out / 'run.json', cfg)
    ref.json_write(out / 'support_manifest.json', {d: [train.get_sample_info(i) for i in indices] for d, indices in manifest.items()})
    ref.json_write(out / 'started.json', dict(seed=seed, from_scratch=True, initial_state_sha256=initial_hash))
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, ref.request_stop)
    print('FRESH_CACHE_TRAIN_START ' + json.dumps(cfg), flush=True)
    best, bad, start_epoch, start_step, old_sums = -1., 0, 0, 0, {}

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
                np.random.seed((ref.SEED + epoch * 1000003 + prior * 1009) % 2**32)
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
                    scaler=float(scaler.get_scale()), execution_revision='fresh_cached_f_multiseed_20261027',
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
            execution_revision='fresh_cached_f_multiseed_20261027')
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
