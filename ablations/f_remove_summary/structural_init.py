"""Match retained weights to the original seed42 RANDOM initialization, never trained weights."""
import gc,hashlib,importlib.util
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
def initialize(model,features):
 spec=importlib.util.spec_from_file_location('F_original_random_reference',ROOT/'reference_model.py')
 reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(42);full=reference.CampaignModel(features,'F')
  full_sha=reference.state_hash(full)
  assert full_sha=='2bbe19389a200f50ad5b52f310f4596f4cce44784935893b084bdda348e77740'
 original=full.state_dict();target=model.state_dict();mapped={};common=0;sliced=[]
 selection=slice(8,9) if not hasattr(model,'dual') else slice(0,8)
 for name,value in target.items():
  old=original[name]
  if old.shape!=value.shape:
   assert name in ('sources.medical_features','sources.shared_codes','sources.task_codes')
   old=old[:,selection];sliced.append(name)
  assert old.shape==value.shape
  mapped[name]=old;common+=old.numel()
 model.load_state_dict(mapped,strict=True)
 assert all(torch.equal(v,mapped[n]) for n,v in model.state_dict().items())
 h=hashlib.sha256()
 for name,tensor in model.state_dict().items():h.update(name.encode());h.update(tensor.contiguous().numpy().tobytes())
 removed=[name for name in original if name not in target]
 assert removed
 result=dict(reference_is_random_initialization=True,trained_checkpoint_loaded=False,reference_full_initial_sha256=full_sha,
    retained_tensors_exactly_match=True,retained_and_sliced_elements=common,sliced_condition_tensors=sliced,removed_state_names=removed,
    structural_initial_sha256=h.hexdigest(),parameters=sum(p.numel() for p in model.parameters()))
 del full,original,mapped;gc.collect()
 return result
