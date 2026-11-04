"""Generate fixed unseen-LV text using only the original frozen local encoders."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'/tank/wk/TS/TSV3')
import torch
from transformers import AutoModel,AutoTokenizer
torch.set_num_threads(4)
prompts=json.loads((ROOT/'target_prompts.json').read_text())
features={};metadata={}
encoder='/tank/wk/TS3D/models/BioLORD-2023-167aab5'
tokenizer=AutoTokenizer.from_pretrained(encoder,local_files_only=True)
model=AutoModel.from_pretrained(encoder,local_files_only=True).cpu().eval()
model.requires_grad_(False)
text=prompts['F']['positive']+prompts['F']['negative']+[prompts['F']['summary']]
with torch.inference_mode():
    e=tokenizer(text,padding=True,truncation=False,return_tensors='pt')
    hidden=model(**e).last_hidden_state
    mask=e['attention_mask'].unsqueeze(-1).to(hidden.dtype)
    features['F']=torch.nn.functional.normalize((hidden*mask).sum(1)/mask.sum(1),dim=-1).cpu()
metadata['F']=dict(encoder='FremyCompany/BioLORD-2023',pooling='attention-mask mean, L2; same as original F')
del model,tokenizer;gc.collect()
try:
    import importlib.util
    spec=importlib.util.spec_from_file_location('original_generate_text','/tank/wk/TS/TSV3/scripts/generate_text_features.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    encoder='dmis-lab/biobert-large-cased-v1.1'
    tokenizer=AutoTokenizer.from_pretrained(encoder,use_fast=False,local_files_only=True)
    model=AutoModel.from_pretrained(encoder,local_files_only=True).cpu().eval();model.requires_grad_(False)
    raw=torch.stack([module.encode_group(prompts['paper_US'][g],tokenizer,model,torch.device('cpu'),96) for g in module.GROUP_ORDER])
    directory=Path('/tank/wk/TS/TSV2/text_features_llm')
    names=['thyroid','TN3K','BUSI_WHU','BUS-BRA','OTU','prostate']
    originals=[]
    for name in names:
        obj=torch.load(directory/('text_features_'+name+'.pt'),map_location='cpu',weights_only=True)
        originals.append(obj['text_features'].float()[:5])
    common=torch.cat(originals).mean(0,keepdim=True)
    features['paper_US']=raw.cpu()-common
    metadata['paper_US']=dict(encoder=encoder,pooling='original five_group_last4_mean_no_special_tokens',
                              centering='fixed mean of original six domains, no target-data fit',domain_embedding='disabled for unseen organ',
                              group_embedding='retained')
except Exception as error:
    metadata['paper_US_unavailable']=dict(type=type(error).__name__,message=str(error))
torch.save(dict(features=features,metadata=metadata,prompts_sha256=hashlib.sha256((ROOT/'target_prompts.json').read_bytes()).hexdigest()),ROOT/'records/target_text.pt')
(ROOT/'records/target_text_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('TEXT_COMPLETE',json.dumps(metadata),{k:list(v.shape) for k,v in features.items()},flush=True)
