"""One fixed-protocol held-out test pass per validation-selected checkpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from common import ROOT as QUEUE,job_named,verify,environment as guarded_environment
parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);args=parser.parse_args()
JOB=job_named(args.job);SEED=JOB['seed'];ROOT=Path(JOB['source'])
OUT = QUEUE / 'results'
VARIANTS = ('A', 'B', 'D', 'E', 'F', 'G')
TEST_COUNTS = {'thyroid': 286, 'TN3K': 614, 'BUSI_WHU': 186, 'BUS-BRA': 283, 'OTU': 382, 'prostate': 16}
METRICS = ('dice', 'iou', 'hd95')
FORBIDDEN_UUID = 'GPU-e8be9cff-d196-2c04-597d-6ed91e4d6468'

def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def environment(physical):
    assert physical==JOB['physical_gpu']
    return guarded_environment(SEED)

def verify_sources():
    verify()
    for name,h in json.loads((ROOT/'bundle_manifest.json').read_text())['files'].items():
        assert sha(ROOT/name)==h,('Source changed',name)


def worker(variant, physical):
    # Before importing torch, check the real physical mapping and inherited UUID.
    expected_env = environment(physical)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected_env['CUDA_VISIBLE_DEVICES']:
        raise RuntimeError('Worker inherited a different GPU UUID')
    verify_sources()
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from campaign import train as ref
    torch.set_num_threads(2)
    ref.set_seed(SEED, deterministic=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one explicitly approved GPU is required')
    device = torch.device('cuda:0')
    directory = OUT / JOB['id']
    started = time.monotonic()
    write(directory / 'status.json', dict(stage='loading_checkpoint', time=now(), variant=variant,
          physical_gpu=physical, uuid=expected_env['CUDA_VISIBLE_DEVICES']))
    run = ROOT / 'runs' / variant / ('seed_'+str(SEED))
    ck_path = run / 'best.pt'
    ck_sha = sha(ck_path)
    ck = torch.load(ck_path, map_location='cpu', weights_only=True)
    config = ck['config']
    original = json.loads((run / 'run.json').read_text())
    history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
    selected = max(history, key=lambda r: r['macro_dice'])
    if ck['epoch_index'] != selected['epoch'] or abs(ck['best_val_dice'] - selected['macro_dice']) > 1e-12:
        raise RuntimeError('Checkpoint is not the validation-selected best epoch')
    for name in ('variant', 'seed', 'support_manifest', 'source_sha256', 'text_bank_sha256', 'train_data_sha256', 'val_data_sha256'):
        if config[name] != original[name]:
            raise RuntimeError('Checkpoint/run metadata mismatch: ' + name)
    if config['variant'] != variant or config['seed'] != SEED:
        raise RuntimeError('Wrong model variant or seed')
    if config['source_sha256'] != sha(ROOT / 'bundle_manifest.json') or config['text_bank_sha256'] != sha(ROOT / 'campaign/text_bank.pt'):
        raise RuntimeError('Checkpoint source/text fingerprint mismatch')
    model = ref.CampaignModel(ref.load_bank()['features'], variant)
    model.load_state_dict(ck['model'], strict=True)
    saved_bank = ck['prototype_bank']
    del ck
    model.to(device).eval()
    roots = {d: str(Path(ref.DATA) / d) for d in ref.DOMAIN_NAMES}
    kw = dict(image_transform=ref.get_image_transform(), image_size=(512, 512), seed=SEED,
              cache_in_memory=True, use_cpu_clahe=False)
    train = ref.MultiOrganDataset(roots, split='train', **kw)
    val = ref.MultiOrganDataset(roots, split='val', **kw)
    for ds, name, counts in ((train, 'train', ref.TRAIN_COUNTS), (val, 'val', ref.VAL_COUNTS)):
        if {d: len(ds.organ_to_indices[d]) for d in ref.DOMAIN_NAMES} != counts:
            raise RuntimeError('Dataset count mismatch: ' + name)
        if ref.data_fingerprint(ds) != config[name + '_data_sha256']:
            raise RuntimeError('Dataset fingerprint mismatch: ' + name)
    manifest = config['support_manifest']
    if ref.manifest_for(train) != manifest:
        raise RuntimeError('Fixed four-support selection differs from training')
    expected_info = json.loads((run / 'support_manifest.json').read_text())
    actual_info = json.loads(json.dumps({d: [train.get_sample_info(i) for i in ids] for d, ids in manifest.items()}))
    if actual_info != expected_info:
        raise RuntimeError('Support sample identities changed')
    bank = ref.prototypes(model, train, manifest, device)
    bank_max_delta = max(float((bank[d].cpu() - saved_bank[d].cpu()).abs().max()) for d in bank)
    if bank_max_delta > 1e-6:
        raise RuntimeError('Rebuilt support prototype differs from checkpoint: ' + str(bank_max_delta))
    del saved_bank

    def metrics(pred, target):
        return ref.binary_dice(pred, target), ref.binary_iou(pred, target), ref.binary_hd95(pred, target)

    @torch.inference_mode()
    def evaluate(ds, split):
        per_domain, cases = {}, []
        with ThreadPoolExecutor(max_workers=2) as pool:
            for domain in ref.DOMAIN_NAMES:
                indices = ds.organ_to_indices[domain]
                futures = []
                for first in range(0, len(indices), 4):
                    chosen = indices[first:first + 4]
                    pairs = [ds[i] for i in chosen]
                    images = torch.stack([p[0] for p in pairs]).to(device)
                    ids = model.ids([domain] * len(chosen), device)
                    result = model.segment(images, bank[domain].to(device), ids)
                    prediction = (result['pred_masks'].softmax(1)[:, 1] >= .5).cpu().numpy().astype(np.uint8)
                    for j, i in enumerate(chosen):
                        target = ds.load_resized_mask(i)
                        futures.append((i, pool.submit(metrics, prediction[j], target)))
                values = []
                for i, future in futures:
                    numbers = future.result()
                    if not all(np.isfinite(n) for n in numbers):
                        raise RuntimeError('Nonfinite evaluation metric')
                    values.append(numbers)
                    cases.append(dict(index=i, domain=domain, sample_id=ds.samples[i]['sample_id'],
                                      **dict(zip(METRICS, numbers))))
                per_domain[domain] = {k: float(np.mean([r[j] for r in values])) for j, k in enumerate(METRICS)}
                write(directory / 'status.json', dict(stage='evaluating_' + split, time=now(), variant=variant,
                      domain=domain, processed=len(cases), total=len(ds), physical_gpu=physical))
                print(split.upper() + '_DOMAIN ' + json.dumps(dict(domain=domain, **per_domain[domain])), flush=True)
        return dict(per_domain=per_domain, counts={d: len(ds.organ_to_indices[d]) for d in ref.DOMAIN_NAMES},
                    **{'macro_' + k: float(np.mean([v[k] for v in per_domain.values()])) for k in METRICS}), cases

    reproduced, _ = evaluate(val, 'val')
    deltas = {k: abs(reproduced['macro_' + k] - selected['macro_' + k]) for k in METRICS}
    per_domain_delta = max(abs(reproduced['per_domain'][d][k] - selected['per_domain'][d][k])
                           for d in ref.DOMAIN_NAMES for k in METRICS)
    if max(*deltas.values(), per_domain_delta) > 1e-9:
        raise RuntimeError('Full validation reproduction failed: ' + str(deltas))
    write(directory / 'validation_reproduction.json', dict(passed=True, epoch=selected['epoch'],
          metrics=reproduced, macro_absolute_deltas=deltas, max_domain_delta=per_domain_delta,
          support_bank_max_abs_delta=bank_max_delta))
    val.clear_cache()
    test = ref.MultiOrganDataset(roots, split='test', **kw)
    if {d: len(test.organ_to_indices[d]) for d in ref.DOMAIN_NAMES} != TEST_COUNTS:
        raise RuntimeError('Held-out test counts do not match the fixed dataset split')
    train_paths = {str(Path(r['image']).resolve()) for r in train.samples}
    test_paths = {str(Path(r['image']).resolve()) for r in test.samples}
    if train_paths & test_paths:
        raise RuntimeError('Support/training paths overlap the test query paths')
    test_hash = ref.data_fingerprint(test)
    test_result, cases = evaluate(test, 'test')
    if sha(ck_path) != ck_sha:
        raise RuntimeError('Checkpoint changed while evaluating')
    with (directory / 'per_case.jsonl').open('x') as f:
        for case in cases:
            f.write(json.dumps(case, allow_nan=False) + '\n')
    result = dict(time=now(), variant=variant, seed=SEED, split='test', image_size=[512, 512],
                  metric_size=[512, 512], threshold=.5, postprocessing=False, inference_precision='float32',
                  aggregation='per-case mean within domain; equal-weight macro mean across six domains',
                  selected_by='validation macro Dice only', checkpoint=str(ck_path), checkpoint_sha256=ck_sha,
                  selected_epoch=selected['epoch'], best_validation_dice=selected['macro_dice'],
                  test_data_sha256=test_hash, support_manifest=manifest, physical_gpu=physical,
                  gpu_uuid=expected_env['CUDA_VISIBLE_DEVICES'], source_manifest_sha256=sha(ROOT / 'bundle_manifest.json'),
                  evaluation_source_sha256=sha(__file__), validation_reproduced=True,
                  seconds=time.monotonic() - started, **test_result)
    write(directory / 'result.json', result)
    write(directory / 'complete.json', dict(time=now(), stage='completed', cases=len(cases),
          result='result.json', checkpoint_unchanged=True))
    print('TEST_COMPLETE ' + json.dumps(result), flush=True)

if __name__=='__main__':
    directory=OUT/JOB['id'];directory.mkdir(parents=True,exist_ok=False)
    try:worker('F',JOB['physical_gpu'])
    except BaseException as error:
        write(directory/'failed.json',dict(time=now(),error=str(error),traceback=traceback.format_exc()))
        raise
