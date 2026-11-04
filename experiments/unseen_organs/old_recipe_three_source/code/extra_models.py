"""Original-source adapters and per-model preprocessing; no target tuning."""
import hashlib,importlib.util,json,sys,os
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/tank/wk/TS/TSV3')
ASSETS=Path('/tank/wk/TS/TSV2/pretrained')
sys.path.insert(0,str(SOURCE))
METHODS=('unetpp','swinunet_official','clipseg_official','lvit_native224','sammed2d_promptfree')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()

def construct(method):
 checkpoint=Path(os.environ['USTS_SOURCE_CKPT'])
 ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
 meta=dict(method=method,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
           input_policy='raw_grayscale_0_1' if method not in METHODS[:2] else 'source_zscore_clamp5_divide5',
           internal_resolution=224 if method=='lvit_native224' else 512,metric_resolution=512)
 if method=='clipseg_official':
  from transformers import CLIPSegConfig,CLIPSegForImageSegmentation
  model=CLIPSegForImageSegmentation(CLIPSegConfig.from_pretrained(str(ASSETS/'clipseg-rd64-refined'),local_files_only=True))
  model.load_state_dict(ck['model_state_dict'],strict=True)
  assert ck['train_mode']=='decoder'
  for p in model.parameters():p.requires_grad=False
  for p in model.decoder.parameters():p.requires_grad=True
  meta['trainable_policy']='original decoder-only; CLIP vision/text frozen'
 elif method=='lvit_native224':
  from tasksegment.models.lvit_official import OfficialLViTAdapter
  cfg=dict(ck['model_config']);cfg['official_repository_path']=str(SOURCE/'third_party/LViT')
  model=OfficialLViTAdapter(**cfg);model.load_state_dict(ck['model_state_dict'],strict=True)
  meta['trainable_policy']='all original LViT trainable parameters; frozen external BERT'
 elif method=='sammed2d_promptfree':
  from tasksegment.models.sammed2d_official import OfficialSAMMed2DAdapter
  cfg=dict(ck['model_config']);cfg.update(official_repository_path=str(SOURCE/'third_party/SAM-Med2D'),base_checkpoint=str(ASSETS/'sam-med2d/sam-med2d_b.pth'))
  assert sha(cfg['base_checkpoint'])==ck['base_checkpoint_sha256']
  model=OfficialSAMMed2DAdapter(**cfg)
  expected={n for n,p in model.network.named_parameters() if p.requires_grad}
  assert expected==set(ck['trainable_delta']),'Source delta must cover every original trainable parameter'
  model.load_trainable_state_dict(ck['trainable_delta'])
  meta.update(trainable_policy='original image encoder adapters, no-mask embedding and mask decoder',
              base_checkpoint_sha256=ck['base_checkpoint_sha256'],delta_strict_coverage=True,
              base_load_report=model.load_report,external_medical_pretraining_exposure='not certified absent')
 else:
  spec=importlib.util.spec_from_file_location('supplement_baseline_predict',SOURCE/'scripts/predict_baseline.py')
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  cls=module.import_model_class(method);cfg=module.sanitize_model_config(cls,ck['model_config'])
  if 'official_repository_path' in cfg:cfg['official_repository_path']=str(SOURCE/'third_party/Swin-Unet')
  model=cls(**cfg);model.load_state_dict(ck['model_state_dict'],strict=True)
  meta['trainable_policy']='all original trainable parameters'
 meta['strict_source_loading']=True
 meta['parameters']=sum(p.numel() for p in model.parameters())
 meta['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
 return model,ck,meta

class LanguageWrapper(nn.Module):
 def __init__(self,model,method,feature,ck):
  super().__init__();self.model=model;self.method=method
  self.register_buffer('feature',feature.detach().clone())
  if method=='clipseg_official':
   self.register_buffer('image_mean',torch.tensor(ck['image_mean']).view(1,3,1,1))
   self.register_buffer('image_std',torch.tensor(ck['image_std']).view(1,3,1,1))
 def train(self,mode=True):
  super().train(mode)
  if self.method=='clipseg_official':self.model.clip.eval()
  return self
 def forward(self,x):
  if self.method=='lvit_native224':return self.model(x,self.feature.expand(len(x),-1,-1))
  pixels=(x.repeat(1,3,1,1).clamp(0,1)-self.image_mean)/self.image_std
  logits=self.model(pixel_values=pixels,conditional_embeddings=self.feature.expand(len(x),-1),interpolate_pos_encoding=True).logits
  if logits.ndim==2:logits=logits[None]
  logits=F.interpolate(logits[:,None].float(),size=(512,512),mode='bilinear',align_corners=False)[:,0]
  return torch.stack([torch.zeros_like(logits),logits],dim=1)

class SAMWrapper(nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,x):return self.model.forward_prompt_free(x)

def load_model(method,organ):
 model,ck,meta=construct(method)
 if method in ('clipseg_official','lvit_native224'):
  text=torch.load(Path(os.environ['USTS_EXTRA_TEXT']),map_location='cpu',weights_only=True)
  model=LanguageWrapper(model,method,text[method][organ],ck)
  meta.update(target_prompt=text['prompts'][organ],text_asset_sha256=sha(Path(os.environ['USTS_EXTRA_TEXT'])))
 elif method=='sammed2d_promptfree':model=SAMWrapper(model)
 del ck
 return model,meta

def prepare():
 assert __import__('os').environ.get('CUDA_VISIBLE_DEVICES')==''
 from transformers import AutoTokenizer,AutoModel
 torch.set_num_threads(2)
 prompts=json.loads((ROOT/'prompts.json').read_text())
 output={'prompts':prompts};audit={}
 for method in METHODS:
  model,ck,meta=construct(method);model.eval()
  if method=='clipseg_official':
   tokenizer=AutoTokenizer.from_pretrained(str(ASSETS/'clipseg-rd64-refined'),local_files_only=True)
   with torch.no_grad():
    tokens=tokenizer(list(prompts.values()),padding=True,truncation=True,return_tensors='pt')
    features=model.get_conditional_embeddings(batch_size=len(prompts),input_ids=tokens['input_ids'],attention_mask=tokens.get('attention_mask'))
   output[method]={o:features[i:i+1].detach().clone() for i,o in enumerate(prompts)}
  elif method=='lvit_native224':
   tokenizer=AutoTokenizer.from_pretrained(str(ASSETS/'bert-base-uncased'),local_files_only=True)
   encoder=AutoModel.from_pretrained(str(ASSETS/'bert-base-uncased'),local_files_only=True).eval()
   source=ck['prompts'];texts=list(source.values())+list(prompts.values())
   tokens=tokenizer(texts,padding='max_length',truncation=True,max_length=10,return_tensors='pt')
   with torch.no_grad():features=encoder(**tokens).last_hidden_state*tokens['attention_mask'].unsqueeze(-1)
   errors={o:float((features[i:i+1]-ck['text_embeddings'][o]).abs().max()) for i,o in enumerate(source)}
   assert max(errors.values())<1e-4,errors
   meta['source_text_reproduction_max_absolute_error']=max(errors.values())
   output[method]={o:features[len(source)+i:len(source)+i+1].detach().clone() for i,o in enumerate(prompts)}
   meta['target_tokens']={o:tokenizer.convert_ids_to_tokens(tokens['input_ids'][len(source)+i]) for i,o in enumerate(prompts)}
   del encoder
  audit[method]=meta;print('CPU_LOAD_OK',method,meta['trainable_parameters'],flush=True)
  del model,ck
  __import__('gc').collect()
 (ROOT/'records').mkdir(exist_ok=True)
 torch.save(output,Path(os.environ['USTS_EXTRA_TEXT']))
 audit['text_asset_sha256']=sha(Path(os.environ['USTS_EXTRA_TEXT']))
 (ROOT/'records/cpu_assets_verified.json').write_text(json.dumps(audit,indent=2)+'\n')
 print('CPU_ASSETS_VERIFIED',flush=True)

if __name__=='__main__':prepare()
