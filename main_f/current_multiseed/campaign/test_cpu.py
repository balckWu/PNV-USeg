"""Small CPU checks of information routing, gradients, and checkpoint identity."""
import gc
import json
import os
from pathlib import Path
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

from campaign.model import CampaignModel, VARIANTS, state_hash
from campaign.train import loss_terms, load_bank, ROOT, json_write


def main():
    torch.set_num_threads(2)
    features = load_bank()['features']
    generator = torch.Generator().manual_seed(17)
    xs = torch.randn(2, 1, 32, 32, generator=generator)
    ys = torch.zeros(2, 1, 32, 32); ys[:, :, 8:24, 8:24] = 1
    xq = torch.randn(1, 1, 32, 32, generator=generator)
    target = ys[:1, 0].long()
    records = {}
    hashes = []
    for variant in VARIANTS:
        torch.manual_seed(42)
        model = CampaignModel(features, variant)
        hashes.append(state_hash(model))
        model.train()
        output = model(xs, ys, xq, torch.tensor([0]))
        loss, _ = loss_terms(output, target)
        assert torch.isfinite(loss)
        loss.backward()
        assert model.dual.image_projection.weight.grad is not None
        assert model.dense.map_head.weight.grad is not None
        gate_grad = model.decoder.up4.skip_gate[0].weight.grad
        assert (gate_grad is None) == (variant == 'G')
        assert not model.sources.medical_features.requires_grad
        assert model.sources.medical_features.grad is None
        if variant == 'B': assert model.sources.task_codes.grad is not None
        else: assert model.sources.task_codes.grad is None
        if variant in ('A', 'D', 'E'): assert model.sources.shared_codes.grad is not None
        if variant in ('F', 'G'): assert model.sources.shared_codes.grad is None
        active_parameters = sum(p.numel() for p in model.parameters() if p.grad is not None)
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (variant, name)
        model.zero_grad(set_to_none=True)
        ids = torch.tensor([0, 5])
        dual, summary = model.sources(ids, variant)
        if variant == 'A':
            assert torch.equal(dual[0], dual[1]) and torch.equal(summary[0], summary[1])
        if variant == 'D':
            assert torch.equal(summary[0], summary[1]) and not torch.equal(dual[0], dual[1])
        if variant == 'E':
            assert torch.equal(dual[0], dual[1]) and not torch.equal(summary[0], summary[1])
        model.eval()
        with torch.inference_mode():
            before = model(xs, ys, xq, torch.tensor([0]))['pred_masks'].clone()
            if variant in ('F', 'G'):
                saved = model.sources.medical_features.clone()
                model.sources.medical_features[0].copy_(saved[5])
                after = model(xs, ys, xq, torch.tensor([0]))['pred_masks']
                difference = float((before - after).abs().max())
                assert difference > 1e-7
                model.sources.medical_features.copy_(saved)
            else: difference = None
        if variant == 'F':
            f_output = before
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / 'state.pt'
                torch.save(model.state_dict(), p)
                model.load_state_dict(torch.load(p, weights_only=True), strict=True)
                with torch.inference_mode():
                    assert torch.equal(before, model(xs, ys, xq, torch.tensor([0]))['pred_masks'])
        if variant == 'G': assert not torch.equal(f_output, before)
        records[variant] = dict(loss=float(loss.detach()), active_parameters=active_parameters,
                                source_replacement_max_difference=difference,
                                channel_gate_gradient=gate_grad is not None)
        del model, output, loss, before, dual, summary
        gc.collect()
    assert len(set(hashes)) == 1, 'Common initialization differs across variants'
    result = dict(passed=True, identical_initial_state_sha256=hashes[0], variants=records)
    json_write(ROOT / 'records/cpu_checks.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
