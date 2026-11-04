import argparse,csv,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from common import *
def main(job):
 expected=environment(job['seed']);assert os.environ.get('CUDA_VISIBLE_DEVICES')==expected['CUDA_VISIBLE_DEVICES'];verify()
 source=Path(job['source'])
 for name,h in read(source/'source_manifest.json')['files'].items():assert sha(source/name)==h
 sys.path.insert(0,str(source))
 import torch,numpy as np
 from tasksegment.checkpoint import load_checkpoint_model
 from tasksegment.data.datasets import MultiOrganDataset,get_image_transform,set_seed
 from tasksegment.training.prototypes import rebuild_visual_prototype_bank
 from tasksegment.training.evaluation import get_task_tokens
 from tasksegment.training.metrics import binary_dice,binary_iou,binary_hd95
 from tasksegment.configs import DOMAIN_NAMES
 torch.set_num_threads(2);set_seed(job['seed'],deterministic=True);assert torch.cuda.device_count()==1
 run=Path(job['run']);ck_path=run/'checkpoint.pth';ck_sha=sha(ck_path)
 model,bank,cfg,ck=load_checkpoint_model(ck_path,torch.device('cpu'))
 assert model.remove_text_branch and not model.requires_condition_inputs
 assert ck['train_config']['seed']==job['seed'] and ck['train_config']['text_ablation_revision']==2
 assert not ck['train_config']['validation_use_postprocess']
 assert not ck['train_config']['prototype_bank_blending_enabled']
 assert ck['train_config']['prototype_eval_support_samples']==4
 assert all(float(v)==.5 for v in ck['train_config']['domain_thresholds'].values())
 rows=list(csv.DictReader((run/'checkpoint.pth.history.csv').open()))
 best=max(rows,key=lambda r:float(r['val_macro_dice']))
 assert ck['epoch']==int(best['epoch']) and abs(ck['best_val_dice']-float(best['val_macro_dice']))<1e-12
 original_manifest=ck['prototype_manifest'];del ck
 model.cuda().eval();device=torch.device('cuda:0')
 roots={d:'/tank/wk/TS/TSV3/data/'+d for d in DOMAIN_NAMES}
 kw=dict(image_transform=get_image_transform(),image_size=(512,512),seed=job['seed'],cache_in_memory=True,use_cpu_clahe=False)
 train=MultiOrganDataset(roots,split='train',**kw)
 splits=read(ROOT/'reference/splits.json')
 for d in DOMAIN_NAMES:assert [train.samples[i]['sample_id'] for i in train.organ_to_indices[d]]==[r['sample_id'] for r in splits[d]['train']]
 rebuilt,manifest=rebuild_visual_prototype_bank(model,train,device,samples_per_domain=4,batch_size=4,seed=job['seed'])
 assert manifest==original_manifest
 bank_delta=max(float((rebuilt.get(d,device=device)-bank.get(d,device=device)).abs().max()) for d in DOMAIN_NAMES)
 assert bank_delta<1e-6
 train.clear_cache();del train
 destination=ROOT/'results'/job['id'];destination.mkdir(parents=True,exist_ok=False)
 def metrics(pred,target):return binary_dice(pred,target),binary_iou(pred,target),binary_hd95(pred,target)
 def evaluate(split):
  dataset=MultiOrganDataset(roots,split=split,**kw);cases=[];by={}
  with torch.inference_mode(),ThreadPoolExecutor(max_workers=2) as pool:
   for d in DOMAIN_NAMES:
    indices=dataset.organ_to_indices[d]
    assert [dataset.samples[i]['sample_id'] for i in indices]==[r['sample_id'] for r in splits[d][split]]
    pending=[]
    for i in indices:
     image,mask,_=dataset[i];x=image[None].cuda()
     task=get_task_tokens(model,bank,{},d,device)
     result=model.segment_with_task(query_feats=model.encode_query(x),task_tokens=task,output_size=x.shape[-2:])
     pred=(result['pred_masks'].softmax(1)[0,1]>=.5).cpu().numpy().astype(np.uint8)
     pending.append((i,pool.submit(metrics,pred,mask.numpy().astype(np.uint8))))
    values=[]
    for i,future in pending:
     numbers=future.result();assert all(np.isfinite(v) for v in numbers)
     values.append(numbers);cases.append(dict(domain=d,sample_id=dataset.samples[i]['sample_id'],**dict(zip(('dice','iou','hd95'),numbers))))
    by[d]={m:float(np.mean([v[k] for v in values])) for k,m in enumerate(('dice','iou','hd95'))}
    write(destination/'status.json',dict(time=now(),stage=split,domain=d,processed=len(cases)))
  dataset.clear_cache()
  return dict(per_domain=by,counts={d:len(dataset.organ_to_indices[d]) for d in DOMAIN_NAMES},**{'macro_'+m:float(np.mean([v[m] for v in by.values()])) for m in ('dice','iou','hd95')}),cases
 start=time.monotonic();validation,_=evaluate('val')
 delta={m:abs(validation['macro_'+m]-float(best['val_macro_'+m])) for m in ('dice','iou','hd95')}
 assert max(delta.values())<1e-9,('Validation reproduction failed',delta)
 write(destination/'validation_reproduction.json',dict(passed=True,differences=delta,support_bank_max_abs_delta=bank_delta))
 test,cases=evaluate('test');assert len(cases)==1767 and sha(ck_path)==ck_sha
 with (destination/'per_case.jsonl').open('x') as f:
  for row in cases:f.write(json.dumps(row)+'\n')
 result=dict(time=now(),model='complete_no_text_v2',seed=job['seed'],split='test',checkpoint=str(ck_path),checkpoint_sha256=ck_sha,
   selected_epoch=int(best['epoch']),selected_by='validation macro Dice',inference_precision='float32',image_size=[512,512],metric_size=[512,512],
   threshold=.5,postprocessing=False,source_manifest_sha256=sha(source/'source_manifest.json'),evaluation_source_sha256=sha(__file__),
   validation_reproduced=True,support_manifest=manifest,physical_gpu=4,gpu_uuid=expected['CUDA_VISIBLE_DEVICES'],seconds=time.monotonic()-start,**test)
 write(destination/'result.json',result);write(destination/'complete.json',dict(time=now(),passed=True,cases=len(cases)));print('TEST_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);args=p.parse_args()
 try:main(job_named(args.job))
 except BaseException as e:write(ROOT/'records'/('failed_'+args.job+'.json'),dict(time=now(),error=str(e),traceback=traceback.format_exc()));raise
