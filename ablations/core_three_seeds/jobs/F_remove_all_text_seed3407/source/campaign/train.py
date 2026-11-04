import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

from tasksegment.configs import DOMAIN_NAMES, DEFAULT_LOSS_CONFIG
from tasksegment.data.datasets import MultiOrganDataset, get_image_transform, set_seed
from tasksegment.training.augmentations import augment_episode_medical
from tasksegment.training.losses import segmentation_loss, deep_supervision_loss, dense_guidance_loss
from tasksegment.training.metrics import binary_dice, binary_iou, binary_hd95
from tasksegment.training.trainer import _set_cosine_lr, _autocast, _make_grad_scaler
from campaign.model import CampaignModel, VARIANTS, state_hash

ROOT = Path(__file__).resolve().parents[1]
SEED = int(os.environ.get('USTS_TRAIN_SEED', '42'))
DATA = '/tank/wk/TS/TSV3/data'
MODALITY = {'thyroid': 'thyroid', 'TN3K': 'thyroid', 'BUSI_WHU': 'BUSI', 'BUS-BRA': 'BUSI'}
TRAIN_COUNTS = dict(zip(DOMAIN_NAMES, [855, 2447, 555, 1312, 656, 67]))
VAL_COUNTS = dict(zip(DOMAIN_NAMES, [285, 432, 186, 280, 164, 14]))
STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True
    print('STOP_REQUESTED: checkpoint at next optimizer boundary', flush=True)


def json_write(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_bank():
    # F visual-only ablation never loads production text features.
    return {'features': None}


def datasets():
    roots = {d: str(Path(DATA) / d) for d in DOMAIN_NAMES}
    kw = dict(image_transform=get_image_transform(), image_size=(512, 512), seed=SEED,
              cache_in_memory=True, use_cpu_clahe=False)
    train = MultiOrganDataset(roots, split='train', **kw)
    val = MultiOrganDataset(roots, split='val', **kw)
    for ds, counts in ((train, TRAIN_COUNTS), (val, VAL_COUNTS)):
        actual = {d: len(ds.organ_to_indices[d]) for d in DOMAIN_NAMES}
        if actual != counts:
            raise ValueError(('Dataset count differs', actual, counts))
    return train, val


def manifest_for(ds):
    rng = np.random.default_rng(SEED)
    return {d: np.sort(rng.choice(np.asarray(ds.organ_to_positive_indices[d]), size=4, replace=False)).tolist()
            for d in sorted(ds.organ_to_positive_indices)}


def data_fingerprint(ds):
    h = hashlib.sha256()
    for row in ds.samples:
        for k in ('image', 'mask'):
            p = Path(row[k]); s = p.stat()
            h.update(f'{p}|{s.st_size}|{s.st_mtime_ns}'.encode())
    return h.hexdigest()


@contextmanager
def data_rng(seed):
    py = random.getstate()
    np_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[0] if torch.cuda.is_available() else []):
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


def episode(ds, domain, epoch, step, device):
    seed = SEED + epoch * 1000003 + step * 1009
    with data_rng(seed):
        support, query = ds.sample_episode_indices(domain, 4, 4)
        if set(support) & set(query):
            raise ValueError('Support/query overlap')
        xs = torch.stack([ds[i][0] for i in support]).to(device)
        ys = torch.stack([ds[i][1] for i in support]).unsqueeze(1).float().to(device)
        xq = torch.stack([ds[i][0] for i in query]).to(device)
        yq = torch.stack([ds[i][1] for i in query]).unsqueeze(1).float().to(device)
        modality = MODALITY.get(domain, 'generic')
        xs, ys = augment_episode_medical(xs, ys, modality)
        xq, yq = augment_episode_medical(xq, yq, modality)
    return xs, ys, xq, yq.squeeze(1).long(), support, query


def domain_schedule(epoch):
    rng = random.Random(SEED + epoch * 1000003)
    result = []
    for _ in range(250):
        chunk = sorted(DOMAIN_NAMES)
        rng.shuffle(chunk)
        result.extend(chunk)
    return result


def loss_terms(output, target):
    main=segmentation_loss(output['pred_masks'],target,DEFAULT_LOSS_CONFIG)
    aux=deep_supervision_loss(output['aux_logits'],target,DEFAULT_LOSS_CONFIG)
    dense=main.new_zeros(())
    semantic=main.new_zeros(())
    return main+aux,dict(main=main,aux=aux,dense=dense,semantic=semantic)


@torch.inference_mode()
def prototypes(model, train, manifest, device):
    model.eval()
    bank = {}
    for d, indices in manifest.items():
        xs = torch.stack([train[i][0] for i in indices]).to(device)
        ys = torch.stack([train[i][1] for i in indices]).unsqueeze(1).float().to(device)
        bank[d] = model.support_tokens(xs, ys).detach().cpu()
    return bank


@torch.inference_mode()
def evaluate(model, val, bank, device):
    model.eval()
    per_domain = {}
    for d in DOMAIN_NAMES:
        values = {'dice': [], 'iou': [], 'hd95': []}
        indices = val.organ_to_indices[d]
        for start in range(0, len(indices), 4):
            selected = indices[start:start + 4]
            xs = torch.stack([val[i][0] for i in selected]).to(device)
            ids = model.ids([d] * len(selected), device)
            output = model.segment(xs, bank[d].to(device), ids)
            pred = (output['pred_masks'].softmax(1)[:, 1] >= .5).cpu().numpy().astype(np.uint8)
            for j, i in enumerate(selected):
                target = val.load_resized_mask(i)
                values['dice'].append(binary_dice(pred[j], target))
                values['iou'].append(binary_iou(pred[j], target))
                values['hd95'].append(binary_hd95(pred[j], target))
        per_domain[d] = {k: float(np.mean(v)) for k, v in values.items()}
        print('VAL_DOMAIN ' + json.dumps(dict(domain=d, **per_domain[d])), flush=True)
    return dict(per_domain=per_domain, counts=VAL_COUNTS,
                **{f'macro_{k}': float(np.mean([v[k] for v in per_domain.values()])) for k in ('dice', 'iou', 'hd95')})


def save_checkpoint(path, model, optimizer, scaler, cfg, epoch_index, next_step,
                    best, bad, sums, bank=None):
    obj = dict(schema=1, architecture='US-TextSeg-ConditionSources-20261025', config=cfg,
               model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
               epoch_index=epoch_index, next_step=next_step, best_val_dice=best,
               bad_epochs=bad, sums=sums, prototype_bank=bank or {},
               cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(0),
               python_rng=random.getstate(),numpy_rng=[np.random.get_state()[0],np.random.get_state()[1].tolist(),int(np.random.get_state()[2]),int(np.random.get_state()[3]),float(np.random.get_state()[4])])
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True, choices=VARIANTS)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    variant = args.variant
    expected = json.loads((ROOT / 'campaign/gpu_map.json').read_text())[str(VARIANTS[variant])]
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected or expected.endswith('6468'):
        raise RuntimeError('Physical GPU/UUID guard rejected configuration')
    torch.set_num_threads(2)
    set_seed(SEED, deterministic=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one explicitly allowed GPU required')
    device = torch.device('cuda:0')
    outdir = ROOT / 'runs' / variant / f'seed_{SEED}'
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / 'run.json').exists() and not args.resume:
        raise FileExistsError('Refusing to overwrite experiment ' + variant)
    bank_data = load_bank()
    train, val = datasets()
    manifest = manifest_for(train)
    init_start = time.monotonic()
    # Dataset inspection is CPU-only and never consumes the model RNG stream.
    set_seed(SEED, deterministic=True)
    model = CampaignModel(bank_data['features'], variant)
    init_hash = state_hash(model)
    protocol = json.loads((ROOT / 'campaign/protocol.json').read_text())
    cfg = dict(protocol=protocol, variant=variant, seed=SEED, physical_gpu=VARIANTS[variant],
               gpu_uuid=expected, source_sha256=file_hash(ROOT / 'bundle_manifest.json'),
               text_bank_sha256=file_hash(ROOT / 'campaign/text_bank.pt'),
               train_data_sha256=data_fingerprint(train), val_data_sha256=data_fingerprint(val),
               support_manifest=manifest, initial_state_sha256=init_hash,
               parameters=sum(p.numel() for p in model.parameters()),
               initial_cpu_setup_seconds=time.monotonic() - init_start)
    json_write(outdir / 'run.json', cfg)
    json_write(outdir / 'support_manifest.json',
               {d: [train.get_sample_info(i) for i in selected] for d, selected in manifest.items()})
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scaler = _make_grad_scaler(True)
    best, bad, start_epoch, start_step = -1., 0, 0, 0
    old_sums = {}
    if args.resume:
        ck = torch.load(outdir / 'last.pt', map_location=device, weights_only=True)
        for k in ('variant', 'source_sha256', 'text_bank_sha256', 'train_data_sha256', 'val_data_sha256', 'support_manifest'):
            if ck['config'][k] != cfg[k]:
                raise ValueError('Resume fingerprint mismatch ' + k)
        model.load_state_dict(ck['model'], strict=True)
        optimizer.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        best, bad = ck['best_val_dice'], ck['bad_epochs']
        start_epoch, start_step, old_sums = ck['epoch_index'], ck['next_step'], ck['sums']
        torch.set_rng_state(ck['cpu_rng'].cpu()); torch.cuda.set_rng_state(ck['cuda_rng'].cpu(), 0)
        random.setstate(ck['python_rng'])
        del ck
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    print('TRAIN_START ' + json.dumps(cfg), flush=True)
    for epoch in range(start_epoch, 150):
        lr = _set_cosine_lr(optimizer, epoch, 150, 5e-5, 1e-6, 5)
        sums = old_sums if epoch == start_epoch and start_step else {k: 0. for k in ('loss', 'main', 'aux', 'dense', 'semantic')}
        schedule = domain_schedule(epoch)
        model.train(); optimizer.zero_grad(set_to_none=True)
        first = start_step if epoch == start_epoch else 0
        epoch_start = time.monotonic(); block_start = epoch_start
        sample_hash = hashlib.sha256()
        for step in range(first, 1500):
            domain = schedule[step]
            xs, ys, xq, yq, support, query = episode(train, domain, epoch, step, device)
            sample_hash.update(json.dumps([domain, support, query]).encode())
            ids = model.ids([domain] * 4, device)
            with _autocast(device, True):
                output = model(xs, ys, xq, ids)
                loss, terms = loss_terms(output, yq)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss {variant} {epoch=} {step=}')
            scaler.scale(loss / 4.).backward()
            total = float(loss.detach()); sums['loss'] += total
            for k, v in terms.items(): sums[k] += float(v.detach())
            del output, loss, terms, xs, ys, xq, yq
            update_due = (step + 1) % 4 == 0
            if update_due:
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 20 == 0 or step == first:
                torch.cuda.synchronize()
                now = time.monotonic()
                status = dict(variant=variant, epoch=epoch + 1, step=step + 1, steps_per_epoch=1500,
                              loss=total, elapsed_epoch_seconds=now - epoch_start, lr=lr,
                              allocated_GiB=torch.cuda.memory_allocated() / 2**30,
                              peak_GiB=torch.cuda.max_memory_allocated() / 2**30,
                              scaler=float(scaler.get_scale()))
                json_write(outdir / 'status.json', status)
                print('STEP ' + json.dumps(status), flush=True)
                block_start = now
            if (step + 1) % 300 == 0 or (STOP and update_due):
                save_checkpoint(outdir / 'last.pt', model, optimizer, scaler, cfg,
                                epoch, step + 1, best, bad, sums)
            if STOP and update_due:
                json_write(outdir / 'stopped.json', dict(epoch=epoch + 1, next_step=step + 1))
                return
        torch.cuda.synchronize()
        train_seconds = time.monotonic() - epoch_start
        val_start = time.monotonic()
        visual_bank = prototypes(model, train, manifest, device)
        metrics = evaluate(model, val, visual_bank, device)
        val_seconds = time.monotonic() - val_start
        improved = metrics['macro_dice'] > best
        if improved: best, bad = metrics['macro_dice'], 0
        else: bad += 1
        row = dict(epoch=epoch + 1, **metrics, **{f'train_{k}': v / 1500 for k, v in sums.items()},
                   lr=lr, train_seconds=train_seconds, validation_seconds=val_seconds,
                   sample_sequence_sha256=sample_hash.hexdigest(), best_val_dice=best, bad_epochs=bad)
        json_write(outdir / f'validation_epoch_{epoch+1:03d}.json', row)
        with (outdir / 'history.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
        if improved:
            save_checkpoint(outdir / 'best.pt', model, optimizer, scaler, cfg, epoch + 1, 0, best, bad, {}, visual_bank)
        save_checkpoint(outdir / 'last.pt', model, optimizer, scaler, cfg, epoch + 1, 0, best, bad, {}, visual_bank)
        print('EPOCH ' + json.dumps(row), flush=True)
        start_step = 0
        if bad >= 20 or STOP: break
    final_name = 'stopped.json' if STOP else 'complete.json'
    json_write(outdir / final_name, dict(best_val_dice=best, last_epoch=epoch + 1,
                                       early_stopped=bad >= 20, test_evaluated=False))


if __name__ == '__main__':
    main()
