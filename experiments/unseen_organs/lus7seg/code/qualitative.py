"""Only fixed hashed patients; load final states without any optimization."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/tank/wk/TS/TSV3')
CAMPAIGN=Path('/tank/wk/US-TextSeg/experiments/seed42_text_mechanisms_20261025')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def experiment(organ):
 spec=importlib.util.spec_from_file_location('fixed_'+organ,ROOT/organ/'code/experiment.py')
 e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e);return e
def worker(organ,method):
 e=experiment(organ);expected=e.gpu_env()['CUDA_VISIBLE_DEVICES']
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==expected
 is_f=method.startswith('F_');sys.path.insert(0,str(CAMPAIGN if is_f else SOURCE))
 import numpy as np
 import torch
 import torch.nn.functional as functional
 from PIL import Image
 from tasksegment.data.datasets import get_image_transform,set_seed
 from tasksegment.training.metrics import binary_dice,binary_iou,binary_hd95
 torch.set_num_threads(2);set_seed(42,deterministic=True);device=torch.device('cuda:0')
 root=ROOT/organ;data=root/'data'
 manifest=json.loads((data/'manifest.json').read_text());rows={r['id']:r for r in manifest['rows']}
 registration=json.loads((ROOT/'records/qualitative_registration.json').read_text())
 identities=registration['selected_ids'][organ]
 source=CAMPAIGN/'runs/F/seed_42/best.pt' if is_f else SOURCE/'paper_artifacts/runs/transunet_pretrained/seed_42/checkpoint.pth'
 source_sha=sha(source);ck=torch.load(source,map_location='cpu',weights_only=True)
 if is_f:
  from campaign.model import CampaignModel
  bank=torch.load(CAMPAIGN/'campaign/text_bank.pt',map_location='cpu',weights_only=True)
  model=CampaignModel(bank['features'],'F');model.load_state_dict(ck['model'],strict=True)
  text=torch.load(root/'records/target_text.pt',map_location='cpu',weights_only=True)
  with torch.no_grad():model.sources.medical_features[0].copy_(functional.normalize(text['features']['F'],dim=-1))
 else:
  spec=importlib.util.spec_from_file_location('baseline_predict',SOURCE/'scripts/predict_baseline.py')
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  cls=module.import_model_class('transunet_official');cfg=module.sanitize_model_config(cls,ck['model_config'])
  model=cls(**cfg);model.load_state_dict(ck['model_state_dict'],strict=True)
 del ck
 model_name='F' if is_f else 'transunet_pretrained';mode='noft' if method=='F_noft' else 'adapt'
 run=root/'results'/model_name/mode/'K16_seed42'
 result=json.loads((run/'result.json').read_text());assert result['checkpoint_sha256']==source_sha
 if mode=='adapt':
  ck=torch.load(run/'adapted_state.pt',map_location='cpu',weights_only=True)
  assert ck['protocol_sha256']==result['protocol_sha256'] and ck['training']['successful_optimizer_updates']==100
  model.load_state_dict(ck['model'],strict=True);del ck
 model.to(device).eval();normalize=get_image_transform()
 def batch(ids):
  xs=[];ys=[]
  for identity in ids:
   r=rows[identity];a=np.asarray(Image.open(data/r['image']).convert('L'),dtype=np.float32)/255.
   b=np.asarray(Image.open(data/r['mask']).convert('L'),dtype=np.uint8)
   xs.append(normalize(torch.from_numpy(a.copy())[None]));ys.append(torch.from_numpy(b.copy())[None].float())
  return torch.stack(xs).to(device),torch.stack(ys).to(device)
 original={r['id']:r for r in json.loads((run/'per_case.json').read_text())}
 target=ROOT/'visuals'/organ/method;target.mkdir(parents=True,exist_ok=False);metrics=[]
 with torch.inference_mode():
  proto=None
  if is_f:
   support=manifest['supports']['42']['16'];tokens=[]
   for start in range(0,16,4):
    x,y=batch(support[start:start+4]);tokens.append(model.support_tokens(x,y))
   proto=torch.stack(tokens).mean(0)
  for identity in identities:
   x,y=batch([identity])
   output=model.segment(x,proto,torch.zeros(1,device=device,dtype=torch.long)) if is_f else model(x)
   if isinstance(output,dict):output=output.get('pred_masks',output.get('logits',next(iter(output.values()))))
   if isinstance(output,(list,tuple)):output=output[-1]
   pred=(output.softmax(1)[:,1]>=.5)[0].cpu().numpy().astype(np.uint8);gt=y[0,0].cpu().numpy().astype(np.uint8)
   scores=dict(dice=binary_dice(pred,gt),iou=binary_iou(pred,gt),hd95=binary_hd95(pred,gt))
   if abs(scores['dice']-original[identity]['dice'])>1e-4 or abs(scores['iou']-original[identity]['iou'])>1e-4:
    raise RuntimeError('Qualitative inference does not reproduce registered result')
   Image.fromarray(pred*255).save(target/(identity+'_prediction.png'))
   Image.open(data/rows[identity]['image']).save(target/(identity+'_image.png'))
   Image.fromarray(gt*255).save(target/(identity+'_target.png'))
   metrics.append(dict(id=identity,**scores,registered_dice=original[identity]['dice']))
 assert sha(source)==source_sha
 (target/'metrics.json').write_text(json.dumps(dict(method=method,organ=organ,K=16,support_draw_seed=42,
    cases=metrics,source_sha256=source_sha,selection='fixed hash, never replaced',physical_gpu=6),indent=2)+'\n')
 print('QUALITATIVE_VERIFIED',organ,method,len(metrics),flush=True)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--organ');parser.add_argument('--method');args=parser.parse_args()
 if args.organ:worker(args.organ,args.method);return
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 done=json.loads((ROOT/'records/all_jobs_finished.json').read_text());assert not done['failed'] and len(done['completed'])==24
 outputs=[]
 for organ in ('liver','gallbladder'):
  for method in ('F_noft','F_adapt','TransUNet_adapt'):
   with (ROOT/'logs'/('qualitative_'+organ+'_'+method+'.log')).open('x') as log:
    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--organ',organ,'--method',method],cwd=ROOT,
      env=experiment(organ).gpu_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    code=child.wait()
   if code:raise RuntimeError('Qualitative worker failed '+organ+' '+method)
   outputs.append(dict(organ=organ,method=method))
 (ROOT/'records/qualitative_complete.json').write_text(json.dumps(dict(passed=True,workers=outputs),indent=2)+'\n')
if __name__=='__main__':main()
