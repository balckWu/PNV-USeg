"""CPU structural checks, retained initialization and synthetic gradient audit."""
import os,sys,json,hashlib
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import torch
from campaign import train as ref
from structural_init import initialize
torch.set_num_threads(2);torch.manual_seed(42)
def forbidden_asset_load(*args,**kwargs):raise AssertionError('Visual-only construction/initialization must not load any torch asset')
torch.load=forbidden_asset_load
assert ref.load_bank()=={'features':None}
model=ref.CampaignModel(None,'F');init=initialize(model)
assert not any(hasattr(model,n) for n in ('sources','dual','dense'))
assert not any(n.startswith(('sources.','dual.','dense.')) for n in model.state_dict())
assert not (ROOT/'campaign/text_bank.pt').exists() and not (ROOT/'campaign/prompts.json').exists()
assert not any(any(t in n for t in ['semantic_log_scale','condition_norm','condition_log_scale','ftext','domain_embedding','group_embedding']) for n in model.state_dict())
xs=torch.randn(4,1,64,64);ys=torch.zeros(4,1,64,64);ys[:,:,16:48,16:48]=1
xq=torch.randn(1,1,64,64);target=torch.zeros(1,64,64,dtype=torch.long);target[:,20:44,20:44]=1
model.train();output=model(xs,ys,xq,model.ids(['thyroid'],torch.device('cpu')))
assert set(output)=={'pred_masks','aux_logits'} and output['pred_masks'].shape==(1,2,64,64)
loss,terms=ref.loss_terms(output,target);assert torch.isfinite(loss)
assert all(terms[n].item()==0 and not terms[n].requires_grad for n in ('dense','semantic'))
assert torch.equal(loss,terms['main']+terms['aux']);loss.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
gradients={}
for name,module in [('encoder',model.encoder),('task_encoder',model.task_encoder),('visual_cross_attention',model.decoder.query_to_task),('decoder',model.decoder)]:
    gradients[name]=any(p.grad is not None and p.grad.abs().sum().item()>0 for p in module.parameters());assert gradients[name],name
gate_names=[n for n,p in model.named_parameters() if 'channel_gate' in n or 'task_gate' in n or 'gate' in n]
assert gate_names, 'Expected visual channel gates'
model.eval()
with torch.inference_mode():
    v=model.support_tokens(xs,ys);a=model.segment(xq,v,model.ids(['thyroid'],torch.device('cpu')))['pred_masks'];b=model.segment(xq,v,model.ids(['prostate'],torch.device('cpu')))['pred_masks'];assert torch.equal(a,b)
result=dict(passed=True,cpu_only=True,synthetic_input_size=[64,64],formal_image_size=[512,512],condition='remove_all_text',deleted_modules=['sources','dual','dense','legacy_unused_semantic_and_condition_parameters'],no_text_feature_buffers_or_learned_placeholders=True,no_text_asset_in_production_source=True,torch_asset_loads_blocked_during_model_initialization_and_forward=True,no_text_auxiliary_outputs_or_losses=True,domain_ids_do_not_change_predictions=True,visual_prototype_interaction_and_channel_gates_retained=True,visual_gate_parameter_names=gate_names,finite_gradients=gradients,initialization=init,synthetic_loss=float(loss.detach()))
(ROOT/'records/cpu_structural_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
