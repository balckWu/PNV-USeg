"""Synthetic CPU interface check. This is not an experiment or score reproduction."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition', choices=['full','without_pn','without_dg','visual_only'], default='full')
    args = p.parse_args()
    sources = {'full':'main_f/current_multiseed', 'without_pn':'ablations/f_remove_dual',
               'without_dg':'ablations/f_remove_summary', 'visual_only':'ablations/f_remove_all_text'}
    sys.path.insert(0, str(ROOT/sources[args.condition]))
    import torch
    from campaign.model import CampaignModel
    torch.set_num_threads(2)
    torch.manual_seed(42)
    features = None if args.condition == 'visual_only' else torch.nn.functional.normalize(
        torch.randn(6,9,768, generator=torch.Generator().manual_seed(1042)), dim=-1)
    model = CampaignModel(features, 'F').cpu().eval()
    images = torch.randn(4,1,64,64)
    masks = torch.zeros(4,1,64,64)
    masks[:,:,16:48,16:48] = 1
    query = torch.randn(1,1,64,64)
    with torch.inference_mode():
        output = model(images, masks, query, torch.tensor([0], dtype=torch.long))
    logits = output['pred_masks']
    assert tuple(logits.shape) == (1,2,64,64)
    assert bool(torch.isfinite(logits).all())
    print(json.dumps(dict(passed=True, device='cpu', condition=args.condition,
        parameters=sum(p.numel() for p in model.parameters()), logits_shape=list(logits.shape),
        synthetic_inputs=True, training_performed=False, reported_scores_reproduced=False)))


if __name__ == '__main__':
    main()
