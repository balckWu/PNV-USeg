from concurrent.futures import ThreadPoolExecutor
import copy
import numpy as np
import torch

from campaign import train as ref


def restore(model, optimizer, scaler, checkpoint):
    model.load_state_dict(checkpoint['model'], strict=True)
    # AdamW's CPU step tensors would otherwise alias the input checkpoint and
    # mutate it across repeated diagnostic restores.
    optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    scaler.load_state_dict(checkpoint['scaler'])
    torch.set_rng_state(checkpoint['cpu_rng'].cpu())
    torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu(), 0)
    ref.random.setstate(checkpoint['python_rng'])
    optimizer.zero_grad(set_to_none=True)


def one_episode(model, cache, domain, epoch, step, device, old=False):
    batch = (ref.episode(cache.train, domain, epoch, step, device) if old
             else cache.episode(domain, epoch, step))
    xs, ys, xq, yq, support, query = batch
    ids = model.ids([domain] * 4, device)
    with ref._autocast(device, True):
        output = model(xs, ys, xq, ids)
        loss, terms = ref.loss_terms(output, yq)
    return loss, terms, support, query


@torch.inference_mode()
def prototypes(model, cache, manifest, device):
    model.eval()
    bank = {}
    for domain, indices in manifest.items():
        images, masks = cache.training_batch(indices)
        bank[domain] = model.support_tokens(images, masks).detach()
    return bank


def _metrics(prediction, target):
    return (ref.binary_dice(prediction, target), ref.binary_iou(prediction, target),
            ref.binary_hd95(prediction, target))


@torch.inference_mode()
def evaluate(model, cache, bank, device):
    model.eval()
    per_domain = {}
    cache.stage_validation()
    try:
        # Metric results are consumed in the original sample order. CPU boundary
        # metrics overlap GPU inference without changing their implementation.
        with ThreadPoolExecutor(max_workers=2) as pool:
            for domain in ref.DOMAIN_NAMES:
                futures = []
                indices = cache.val.organ_to_indices[domain]
                for start in range(0, len(indices), 4):
                    selected = indices[start:start + 4]
                    images = cache.validation_batch(selected)
                    ids = model.ids([domain] * len(selected), device)
                    output = model.segment(images, bank[domain], ids)
                    pred = (output['pred_masks'].softmax(1)[:, 1] >= .5).cpu().numpy().astype(np.uint8)
                    for j, index in enumerate(selected):
                        futures.append(pool.submit(_metrics, pred[j], cache.val_masks[index]))
                values = [f.result() for f in futures]
                per_domain[domain] = {name: float(np.mean([value[i] for value in values]))
                                      for i, name in enumerate(('dice', 'iou', 'hd95'))}
                print('VAL_DOMAIN ' + ref.json.dumps(dict(domain=domain, **per_domain[domain])), flush=True)
    finally:
        cache.release_validation()
    return dict(per_domain=per_domain, counts=ref.VAL_COUNTS,
                **{'macro_' + k: float(np.mean([v[k] for v in per_domain.values()]))
                   for k in ('dice', 'iou', 'hd95')})
