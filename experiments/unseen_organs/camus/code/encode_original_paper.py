"""Recover original BioBERT text preprocessing by matching saved SOURCE features."""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'/tank/wk/TS/TSV3')
import torch
from transformers import BertConfig,BertModel,BertTokenizer
torch.set_num_threads(4)

def main():
    assets=ROOT/'assets/biobert_large'
    provenance=json.loads((assets/'provenance.json').read_text())
    for row in provenance['files']:
        h=hashlib.sha256()
        with (assets/row['file']).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=row['sha256']:raise RuntimeError('Encoder asset SHA256 changed')
    spec=importlib.util.spec_from_file_location('original_text_generator','/tank/wk/TS/TSV3/scripts/generate_text_features.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=BertModel(BertConfig.from_json_file(str(assets/'config.json')))
    raw=torch.load(assets/'pytorch_model.bin',map_location='cpu',weights_only=True)
    if 'state_dict' in raw:raw=raw['state_dict']
    if any(k.startswith('bert.') for k in raw):raw={k[5:]:v for k,v in raw.items() if k.startswith('bert.')}
    model.load_state_dict({k:raw[k] for k in model.state_dict()},strict=True)
    del raw;gc.collect();model.eval().requires_grad_(False)
    directory=Path('/tank/wk/TS/TSV2/text_features_llm')
    source=torch.load(directory/'text_features_thyroid.pt',map_location='cpu',weights_only=True)
    checks=[]
    for lower in (True,False):
        tokenizer=BertTokenizer(vocab=str(assets/'vocab.txt'),do_lower_case=lower)
        generated=torch.stack([module.encode_group(source['prompts'][g],tokenizer,model,torch.device('cpu'),96) for g in module.GROUP_ORDER])
        expected=source['text_features'].float()[:5]
        maximum=float((generated-expected).abs().max())
        cosine=float(torch.nn.functional.cosine_similarity(generated,expected).mean())
        checks.append(dict(do_lower_case=lower,max_abs=maximum,mean_cosine=cosine))
    best=min(checks,key=lambda r:r['max_abs'])
    if best['max_abs']>0.01 or best['mean_cosine']<0.99999:
        raise RuntimeError('Original source text features not reproducible: '+str(checks))
    tokenizer=BertTokenizer(vocab=str(assets/'vocab.txt'),do_lower_case=best['do_lower_case'])
    prompts=json.loads((ROOT/'target_prompts.json').read_text())['paper_US']
    target=torch.stack([module.encode_group(prompts[g],tokenizer,model,torch.device('cpu'),96) for g in module.GROUP_ORDER])
    old=[]
    for name in ('thyroid','TN3K','BUSI_WHU','BUS-BRA','OTU','prostate'):
        old.append(torch.load(directory/('text_features_'+name+'.pt'),map_location='cpu',weights_only=True)['text_features'].float()[:5])
    target=target.cpu()-torch.cat(old).mean(0,keepdim=True)
    metadata=dict(encoder=provenance['repo'],revision=provenance['revision'],source_feature_reproduction=checks,
                  matched_do_lower_case=best['do_lower_case'],centering='fixed original six-domain mean',
                  unseen_domain_identity_residual='disabled, weights otherwise strictly loaded',group_residual='retained',
                  prompts_sha256=hashlib.sha256((ROOT/'target_prompts.json').read_bytes()).hexdigest())
    temp=ROOT/'records/paper_target_text.pending.pt'
    torch.save(dict(features={'paper_US':target},metadata=metadata),temp)
    temp.replace(ROOT/'records/paper_target_text.pt')
    (ROOT/'records/paper_target_text_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('ORIGINAL_ENCODER_REPRODUCED',json.dumps(metadata),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        (ROOT/'records/biobert_encoding_failed.json').write_text(json.dumps(dict(type=type(error).__name__,message=str(error)),indent=2)+'\n')
        raise
