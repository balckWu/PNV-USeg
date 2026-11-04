"""Audit the source release using the standard library only; never uses a GPU."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {'.pt','.pth','.ckpt','.safetensors','.bin','.npz','.npy','.pkl','.pickle',
             '.onnx','.h5','.hdf5','.dcm','.nii','.jpg','.jpeg','.bmp','.tif','.tiff',
             '.log','.jsonl','.zip','.gz','.tar','.pyc','.pem','.key','.png','.svg','.pdf','.csv','.tsv'}
SECRETS = [re.compile(r'''(?i)(?:password|passwd|api_key|access_token)\s*=\s*["']([^"'\n]{4,})["']'''),
           re.compile(r'(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{25,})'),
           re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')]
REQUIRED = ['main_f/seed42/cache_v2/engine.py','main_f/current_multiseed/campaign/model.py',
    'main_f/current_multiseed/fresh_train.py','ablations/f_remove_dual/structural_init.py',
    'ablations/f_remove_summary/train_ablation.py','ablations/f_remove_all_text/campaign/model.py',
    'ablations/core_three_seeds/evaluate_f.py','ablations/core_three_seeds/evaluate_visual.py',
    'experiments/unseen_organs/without_pn/code/heart.py',
    'experiments/unseen_organs/without_pn/code/liver.py',
    'experiments/unseen_organs/without_pn/code/gallbladder.py',
    'experiments/unseen_organs/old_recipe_three_source/code/controller.py',
    'experiments/unseen_organs/without_pn_k8/code/dual8_heart.py',
    'figures/core_ablation/plot_ablation.py','figures/plot_radar.py','figures/plot_full_tradeoff.py',
    'configs/paper_protocol.json','docs/REPRODUCE_PAPER.md']
for condition in ['F_remove_dual','F_remove_summary','F_remove_all_text']:
    for seed in [3407,2026]:
        prefix=f'ablations/core_three_seeds/jobs/{condition}_seed{seed}/source/'
        REQUIRED.extend(prefix+n for n in ['campaign/model.py','train_ablation.py','structural_init.py','expected_initialization.json'])
for name in ['unet','unetpp','transunet_official','swinunet_official','clipseg_official','lvit_official','sammed2d_promptfree','iris2d']:
    REQUIRED.append('baselines/original_tsv3/scripts/train_'+name+'.py')


def check():
    errors=[]; verified=0; parsed=0
    for manifest in ['provenance/source_manifest.json','provenance/latest_source_manifest.json']:
        for row in json.loads((ROOT/manifest).read_text(encoding='utf8'))['files']:
            path=(ROOT/row['path']).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                errors.append(dict(kind='missing_or_unsafe_source',path=row['path']));continue
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
                errors.append(dict(kind='source_hash_mismatch',path=row['path']))
            verified+=1
    for rel in REQUIRED:
        if not (ROOT/rel).is_file():errors.append(dict(kind='missing_entry',path=rel))
    files=[p for p in ROOT.rglob('*') if p.is_file() and not set(p.relative_to(ROOT).parts)&{'.git','__pycache__'}]
    for path in files:
        rel=path.relative_to(ROOT).as_posix()
        if path.is_symlink() or path.suffix.lower() in FORBIDDEN or path.name.startswith('.env'):
            errors.append(dict(kind='non_source_or_secret_filename',path=rel))
        if path.suffix.lower() in FORBIDDEN:continue
        text=path.read_text(encoding='utf-8-sig')
        if any(pattern.search(text) for pattern in SECRETS):
            errors.append(dict(kind='credential_candidate',path=rel))
        if path.suffix=='.py':
            try:ast.parse(text, filename=rel);parsed+=1
            except SyntaxError as e:errors.append(dict(kind='syntax',path=rel,line=e.lineno))
        elif path.suffix=='.json':
            try:json.loads(text)
            except ValueError:errors.append(dict(kind='json',path=rel))
    return dict(passed=not errors,files=len(files),bytes=sum(p.stat().st_size for p in files),
                frozen_source_hashes_verified=verified,python_files_parsed=parsed,
                active_entrypoints_checked=len(REQUIRED),errors=errors,
                scope='Source integrity, syntax, entry points and publication exclusions; no training or inference')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path)
    args=p.parse_args();report=check()
    if args.report:
        target=args.report.resolve()
        if target.is_relative_to(ROOT):raise ValueError('Write audit reports outside the source tree')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
