import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
from transformers import AutoModel, AutoTokenizer

from tasksegment.configs import DOMAIN_NAMES


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/tank/wk/TS3D/models/BioLORD-2023-167aab5')
    args = p.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent
    out = root / 'text_bank.pt'
    if out.exists():
        raise FileExistsError(out)
    prompt_path = root / 'prompts.json'
    prompts = json.loads(prompt_path.read_text(encoding='utf-8'))
    sentences = []
    for domain in DOMAIN_NAMES:
        row = prompts['domains'][domain]
        assert len(row['positive']) == len(row['negative']) == 4
        sentences += row['positive'] + row['negative'] + [row['summary']]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).cpu().eval()
    model.requires_grad_(False)
    vectors, lengths = [], []
    with torch.inference_mode():
        for start in range(0, len(sentences), 8):
            text = sentences[start:start + 8]
            lengths += [len(tokenizer(s, truncation=False)['input_ids']) for s in text]
            encoded = tokenizer(text, padding=True, truncation=False, return_tensors='pt')
            hidden = model(**encoded).last_hidden_state
            mask = encoded['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            vectors.append(torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1))
    assert max(lengths) <= 128
    features = torch.cat(vectors).reshape(6, 9, 768).contiguous().clone()
    assert torch.isfinite(features).all()
    source_files = sorted(Path(args.model).glob('*.safetensors')) + sorted(Path(args.model).glob('pytorch_model*.bin'))
    metadata = dict(model_id='FremyCompany/BioLORD-2023', local_model=str(args.model),
                    source_weight_sha256={p.name: digest(p) for p in source_files},
                    prompt_sha256=digest(prompt_path), pooling='attention_mask_mean_l2',
                    domain_names=list(DOMAIN_NAMES), max_token_length=max(lengths),
                    roles=['positive'] * 4 + ['negative'] * 4 + ['summary'])
    assert source_files, 'No encoder weight files found'
    torch.save(dict(features=features, metadata=metadata, prompts=prompts), out)
    stats = {}
    for i, domain in enumerate(DOMAIN_NAMES):
        sim = features[i, :4] @ features[i, 4:8].T
        stats[domain] = dict(positive_negative_mean=float(sim.mean()), minimum=float(sim.min()), maximum=float(sim.max()))
    record = dict(metadata=metadata, bank_sha256=digest(out), shape=list(features.shape), similarity=stats)
    (root / 'text_audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
