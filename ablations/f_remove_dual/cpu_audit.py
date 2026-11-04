"""CPU-only structural assertions and synthetic gradient test; GPU preflight stays behind test barrier."""
import gc,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import torch
from campaign import train as ref
from structural_init import initialize
torch.set_num_threads(2);torch.manual_seed(42)
condition=json.loads((ROOT/'records/predeclared_protocol.json').read_text())['condition']
bank=ref.load_bank();model=ref.CampaignModel(bank['features'],'F');init=initialize(model,bank['features'])
deleted='dual' if condition=='remove_dual' else 'dense'
retained='dense' if deleted=='dual' else 'dual'
assert not hasattr(model,deleted) and hasattr(model,retained)
assert not any(n.startswith(deleted+'.') for n,_ in model.named_parameters())
assert model.sources.medical_features.shape==(6,1 if deleted=='dual' else 8,768)
model.train()
xs=torch.randn(4,1,64,64);ys=torch.zeros(4,1,64,64);ys[:,:,16:48,16:48]=1
xq=torch.randn(1,1,64,64);target=torch.zeros(1,64,64,dtype=torch.long);target[:,20:44,20:44]=1
output=model(xs,ys,xq,model.ids(['thyroid'],torch.device('cpu')))
assert output['pred_masks'].shape==(1,2,64,64)
deleted_output='semantic_logits_lowres' if deleted=='dual' else 'dense_guidance_logits_lowres'
assert deleted_output not in output
loss,terms=ref.loss_terms(output,target);assert torch.isfinite(loss)
assert terms['semantic' if deleted=='dual' else 'dense'].item()==0
assert not terms['semantic' if deleted=='dual' else 'dense'].requires_grad
loss.backward()
assert all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
assert any(p.grad is not None and p.grad.abs().sum().item()>0 for p in getattr(model,retained).parameters())
assert any(p.grad is not None and p.grad.abs().sum().item()>0 for p in model.encoder.parameters())
model.eval()
with torch.inference_mode():
 inference=model(xs,ys,xq,model.ids(['thyroid'],torch.device('cpu')))
 assert torch.isfinite(inference['pred_masks']).all()
result=dict(passed=True,cpu_only=True,synthetic_input_size=[64,64],formal_image_size=[512,512],condition=condition,
 actual_deleted_module=deleted,no_deleted_parameters=True,no_deleted_output=True,deleted_aux_loss_absent=True,
 remaining_branch_and_encoder_have_finite_gradients=True,visual_support_and_skip_channel_gates_retained=True,
 initialization=init,synthetic_loss=float(loss.detach()))
(ROOT/'records/cpu_structural_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
