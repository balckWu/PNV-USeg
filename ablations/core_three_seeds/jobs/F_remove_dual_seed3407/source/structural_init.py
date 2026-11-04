"""Copy only same-seed RANDOM full-model tensors; never load trained weights."""
import os,gc,hashlib,importlib.util,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
def digest(model):
 h=hashlib.sha256()
 for n,v in model.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def initialize(model,features=None):
 seed=int(os.environ['USTS_TRAIN_SEED']);meta=json.loads((ROOT/'expected_initialization.json').read_text())
 assert seed==meta['seed']
 if meta['kind']=='visual':
  assert features is None and not any(hasattr(model,n) for n in ['sources','dual','dense'])
  assert not any(hasattr(model.task_encoder,n) for n in ['semantic_log_scale','condition_norm','condition_log_scale','ftext_proj','ftext_norm','domain_embedding','group_embedding'])
  assert digest(model)==meta['structural_initial_sha256']
 else:
  spec=importlib.util.spec_from_file_location('same_seed_F_random_reference',ROOT/'reference_model.py')
  reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(seed);full=reference.CampaignModel(features,'F')
   assert digest(full)==meta['reference_full_initial_sha256']
  original=full.state_dict();mapped={};selection=slice(8,9) if meta['kind']=='dual' else slice(0,8)
  for n,v in model.state_dict().items():
   t=original[n]
   if t.shape!=v.shape:
    assert n in ['sources.medical_features','sources.shared_codes','sources.task_codes'];t=t[:,selection]
   assert t.shape==v.shape;mapped[n]=t
  model.load_state_dict(mapped,strict=True)
  assert all(torch.equal(v,mapped[n]) for n,v in model.state_dict().items())
  assert digest(model)==meta['structural_initial_sha256']
  del full,original,mapped;gc.collect()
 return dict(reference_is_random_initialization=True,trained_checkpoint_loaded=False,seed=seed,reference_full_initial_sha256=meta['reference_full_initial_sha256'],retained_tensors_exactly_match=True,structural_initial_sha256=meta['structural_initial_sha256'],parameters=meta['parameters'],visual_production_text_asset_loads=0 if meta['kind']=='visual' else None)
