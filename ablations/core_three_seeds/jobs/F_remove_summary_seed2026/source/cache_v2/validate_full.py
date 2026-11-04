import json
import os
import time
import torch
from campaign import train as ref
from cache_v2.input_cache import InputCache
from cache_v2 import engine


def main():
    variant = 'F'
    expected = json.loads((ref.ROOT / 'campaign/gpu_map.json').read_text())['5']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected or expected.endswith('6468'):
        raise RuntimeError('GPU UUID guard failed')
    torch.set_num_threads(2)
    ref.set_seed(42, deterministic=True)
    device = torch.device('cuda:0')
    checkpoint = torch.load(ref.ROOT / 'archives/cache_v2_20261026/F/last.pt', map_location='cpu', weights_only=True)
    train, val = ref.datasets()
    model = ref.CampaignModel(ref.load_bank()['features'], variant).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    cache = InputCache(train, val, device)
    manifest = checkpoint['config']['support_manifest']
    torch.cuda.synchronize(); start = time.monotonic()
    bank = ref.prototypes(model, train, manifest, device)
    original = ref.evaluate(model, val, bank, device)
    old_seconds = time.monotonic() - start
    torch.cuda.synchronize(); start = time.monotonic()
    bank = engine.prototypes(model, cache, manifest, device)
    cached = engine.evaluate(model, cache, bank, device)
    new_seconds = time.monotonic() - start
    result = dict(variant=variant, original=original, cached=cached, exact_equal=original == cached,
                  original_seconds=old_seconds, cached_seconds=new_seconds, cases=sum(ref.VAL_COUNTS.values()))
    ref.json_write(ref.ROOT / 'records/cache_v2_full_validation.json', result)
    if not result['exact_equal']:
        raise AssertionError('Full validation aggregation changed')
    print('FULL_VALIDATION_PASSED ' + json.dumps({k: result[k] for k in ('exact_equal', 'original_seconds', 'cached_seconds', 'cases')}), flush=True)


if __name__ == '__main__':
    main()
