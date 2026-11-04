import gc,hashlib,importlib.util,json,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'/tank/wk/TS/TSV3')
import torch
from transformers import AutoTokenizer,AutoModel,BertConfig,BertModel,BertTokenizer
torch.set_num_threads(4)
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 encoder='/tank/wk/TS3D/models/BioLORD-2023-167aab5'
 tokenizer=AutoTokenizer.from_pretrained(encoder,local_files_only=True)
 model=AutoModel.from_pretrained(encoder,local_files_only=True).cpu().eval().requires_grad_(False)
 for organ in ('liver','gallbladder'):
  root=ROOT/organ;prompts=json.loads((root/'target_prompts.json').read_text())
  strings=prompts['F']['positive']+prompts['F']['negative']+[prompts['F']['summary']]
  with torch.inference_mode():
   e=tokenizer(strings,padding=True,truncation=False,return_tensors='pt');h=model(**e).last_hidden_state
   mask=e['attention_mask'].unsqueeze(-1).to(h.dtype)
   vectors=torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1),dim=-1)
  metadata=dict(F=dict(encoder='FremyCompany/BioLORD-2023',pooling='original attention-mask mean, L2'))
  torch.save(dict(features={'F':vectors},metadata=metadata),root/'records/target_text.pt')
  write(root/'records/target_text_metadata.json',metadata)
 del model,tokenizer;gc.collect()
 assets=ROOT.parent/'assets/biobert_large';provenance=json.loads((assets/'provenance.json').read_text())
 for entry in provenance['files']:
  h=hashlib.sha256()
  with (assets/entry['file']).open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  assert h.hexdigest()==entry['sha256']
 model=BertModel(BertConfig.from_json_file(str(assets/'config.json')))
 raw=torch.load(assets/'pytorch_model.bin',map_location='cpu',weights_only=True)
 raw={k[5:]:v for k,v in raw.items() if k.startswith('bert.')}
 model.load_state_dict({k:raw[k] for k in model.state_dict()},strict=True);del raw;gc.collect();model.eval().requires_grad_(False)
 tokenizer=BertTokenizer(vocab=str(assets/'vocab.txt'),do_lower_case=True)
 spec=importlib.util.spec_from_file_location('original_text_generator','/tank/wk/TS/TSV3/scripts/generate_text_features.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 directory=Path('/tank/wk/TS/TSV2/text_features_llm')
 source=torch.load(directory/'text_features_thyroid.pt',map_location='cpu',weights_only=True)
 generated=torch.stack([module.encode_group(source['prompts'][g],tokenizer,model,torch.device('cpu'),96) for g in module.GROUP_ORDER])
 error=float((generated-source['text_features'].float()[:5]).abs().max());assert error<.01
 mean=torch.cat([torch.load(directory/('text_features_'+n+'.pt'),map_location='cpu',weights_only=True)['text_features'].float()[:5] for n in ('thyroid','TN3K','BUSI_WHU','BUS-BRA','OTU','prostate')]).mean(0,keepdim=True)
 for organ in ('liver','gallbladder'):
  root=ROOT/organ;prompts=json.loads((root/'target_prompts.json').read_text())['paper_US']
  vectors=torch.stack([module.encode_group(prompts[g],tokenizer,model,torch.device('cpu'),96) for g in module.GROUP_ORDER])-mean
  metadata=dict(encoder=provenance['repo'],revision=provenance['revision'],source_feature_reproduction_max_abs=error,
    centering='fixed original six-source-domain mean',unseen_domain_identity_residual='disabled',group_residual='retained',
    prompts_sha256=hashlib.sha256((root/'target_prompts.json').read_bytes()).hexdigest())
  torch.save(dict(features={'paper_US':vectors},metadata=metadata),root/'records/paper_target_text.pt')
  write(root/'records/paper_target_text_metadata.json',metadata)
  print('TEXT_READY',organ,json.dumps(metadata),flush=True)
 write(ROOT/'records/all_text_ready.json',dict(passed=True,source_encoder_reproduction_max_abs=error))
if __name__=='__main__':
 try:main()
 except BaseException as error:
  write(ROOT/'records/text_prepare_failed.json',dict(type=type(error).__name__,message=str(error)));raise
