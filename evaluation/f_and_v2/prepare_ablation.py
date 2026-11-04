"""Freeze two single-sided text controls before seeing their test results."""
import shutil
from common import *
def prepare():
 source=PARENT/'f_extra_seeds_20261027'
 parent_manifest=read(source/'bundle_manifest.json')
 for name,h in parent_manifest['files'].items():assert sha(source/name)==h
 prepared=[]
 for condition in ('positive_only','negative_only'):
  root=ROOT/'ablations'/condition
  assert not root.exists();root.mkdir(parents=True)
  for name in parent_manifest['files']:
   target=root/name;target.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(source/name,target)
  for name in ('parent_bundle_manifest.json','parent_cache_manifest.json'):
   shutil.copy2(source/name,root/name)
  for name in ('records','logs','runs/F/seed_42'):(root/name).mkdir(exist_ok=True,parents=True)
  model=(root/'campaign/model.py').read_text()
  assert model.count('projected = self.image_projection(image)')==1
  selection='banks[:, :1]' if condition=='positive_only' else 'banks[:, 1:]'
  model=model.replace('projected = self.image_projection(image)','projected = self.image_projection(image)\n        banks = '+selection)
  old='margin = class_scores[:, 0] - class_scores[:, 1]';assert model.count(old)==1
  model=model.replace(old,'margin = '+('class_scores[:, 0]' if condition=='positive_only' else '-class_scores[:, 0]'))
  (root/'campaign/model.py').write_text(model)
  train=(root/'fresh_train.py').read_text()
  train=train.replace('choices=(3407, 2026)','choices=(42,)').replace('choices=(4, 5)','choices=(4,)')
  train=train.replace("protocol['training_seed'] = args.seed","protocol['training_seed'] = args.seed\n    protocol['condition_ablation'] = "+repr(condition))
  train=train.replace('initial_hash = ref.state_hash(model)',"initial_hash = ref.state_hash(model)\n    assert initial_hash=='2bbe19389a200f50ad5b52f310f4596f4cce44784935893b084bdda348e77740'")
  train=train.replace("scaler = ref._make_grad_scaler(True)","scaler = ref._make_grad_scaler(True)\n    if args.preflight:scaler=torch.amp.GradScaler('cuda',init_scale=128.)")
  train=train.replace('schedule = ref.domain_schedule(0)','schedule = list(ref.DOMAIN_NAMES)').replace('for step in range(4):','for step in range(6):').replace('preflight_episodes=4','preflight_episodes=6')
  (root/'train_ablation.py').write_text(train)
  campaign=(root/'campaign/train.py').read_text()
  old='python_rng=random.getstate())';assert campaign.count(old)==1
  campaign=campaign.replace(old,"python_rng=random.getstate(), numpy_rng=[np.random.get_state()[0],np.random.get_state()[1].tolist(),int(np.random.get_state()[2]),int(np.random.get_state()[3]),float(np.random.get_state()[4])])")
  (root/'campaign/train.py').write_text(campaign)
  for p in root.rglob('*.py'):compile(p.read_text(),str(p),'exec')
  files={}
  for folder in ('campaign','cache_v2','tasksegment','paperlab'):
   for p in (root/folder).rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts:files[str(p.relative_to(root))]=sha(p)
  for name in ('fresh_train.py','train_ablation.py'):files[name]=sha(root/name)
  write(root/'bundle_manifest.json',dict(name='F_'+condition+'_seed42',parent_source_sha256=sha(source/'bundle_manifest.json'),files=files))
  write(root/'records/predeclared_protocol.json',dict(time=now(),condition=condition,seed=42,label_fraction=1.,from_scratch=True,
       positive_sentence_count=4 if condition=='positive_only' else 0,negative_sentence_count=4 if condition=='negative_only' else 0,
       single_class_topk=2,single_class_score_sign=1 if condition=='positive_only' else -1,
       summary_and_visual_channel_gate_unchanged=True,shared_head_parameters_and_initial_state_unchanged=True,
       loss_and_optimizer_budget_unchanged=True,epochs_max=150,patience=20,episodes_per_epoch=1500,support=4,query=4,grad_accum=4,
       image_size=[512,512],fixed_eval_K=4,threshold=.5,postprocessing=False,selection='validation macro Dice',
       no_test_based_tuning=True,physical_gpu=4))
  prepared.append(dict(id='F_'+condition,condition=condition,source=str(root),source_manifest_sha256=sha(root/'bundle_manifest.json')))
 write(ROOT/'records/ablations_prepared.json',dict(time=now(),jobs=prepared))
 return prepared
if __name__=='__main__':prepare()
