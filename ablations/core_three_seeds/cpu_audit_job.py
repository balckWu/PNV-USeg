import os,sys,json,hashlib,importlib.util,gc
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
ROOT=Path(__file__).resolve().parent
j=json.loads((ROOT/'jobs.json').read_text())[int(sys.argv[1])]
source=Path(j['source']);seed=j['seed'];os.environ['USTS_TRAIN_SEED']=str(seed)
sys.path.insert(0,str(source))
import torch
torch.set_num_threads(2)
def sha(m):
 h=hashlib.sha256()
 for n,v in m.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
refpath=Path(j['full_reference_source'])/'campaign/model.py'
spec=importlib.util.spec_from_file_location('full_reference_cpu',refpath)
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
features=torch.load(Path(j['full_reference_source'])/'campaign/text_bank.pt',map_location='cpu',weights_only=True)['features']
torch.manual_seed(seed);full=reference.CampaignModel(features,'F')
full_sha=sha(full);reference_run=json.loads((ROOT/'reference_runs'/('F_seed'+str(seed)+'.json')).read_text())
assert full_sha==reference_run['initial_state_sha256'],('Full random initialization did not reproduce',full_sha,reference_run['initial_state_sha256'])
from campaign.model import CampaignModel
torch.manual_seed(seed);model=CampaignModel(None if j['kind']=='visual' else features,'F')
original=full.state_dict();mapped={};sliced=[]
for n,v in model.state_dict().items():
 t=original[n]
 if t.shape!=v.shape:
  assert j['kind']!='visual' and n in ['sources.medical_features','sources.shared_codes','sources.task_codes']
  t=t[:,slice(8,9) if j['kind']=='dual' else slice(0,8)];sliced.append(n)
 assert t.shape==v.shape;mapped[n]=t
if j['kind']=='visual':assert all(torch.equal(v,mapped[n]) for n,v in model.state_dict().items()),'Pure visual common initialization differs'
else:model.load_state_dict(mapped,strict=True)
meta=dict(seed=seed,kind=j['kind'],reference_full_initial_sha256=full_sha,structural_initial_sha256=sha(model),parameters=sum(p.numel() for p in model.parameters()),sliced_tensors=sliced,all_retained_tensors_equal_to_same_seed_random_F=True,trained_weights_loaded=False,visual_runtime_reads_no_text_assets=j['kind']=='visual')
(source/'expected_initialization.json').write_text(json.dumps(meta,indent=2))
from structural_init import initialize
audit=initialize(model,None if j['kind']=='visual' else features)
assert audit['structural_initial_sha256']==meta['structural_initial_sha256']
meta['passed']=True
(ROOT/'records'/('cpu_'+j['id']+'.json')).write_text(json.dumps(meta,indent=2))
print(json.dumps(meta))
