"""Verify the visual seed42 random initialization; no text/model asset reads."""
import hashlib,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent

def initialize(model,features=None):
    assert features is None and not any(hasattr(model,n) for n in ('sources','dual','dense'))
    assert not any(hasattr(model.task_encoder,n) for n in ('semantic_log_scale','condition_norm','condition_log_scale','ftext_proj','ftext_norm','domain_embedding','group_embedding'))
    metadata=json.loads((ROOT/'expected_visual_initialization.json').read_text())
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        assert not name.startswith(('sources.','dual.','dense.'))
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    digest=h.hexdigest();assert digest==metadata['expected_initial_sha256'],(digest,metadata)
    return dict(reference_is_random_initialization=True,trained_checkpoint_loaded=False,retained_tensors_exactly_match_F_random_initialization=True,initialization_evidence='Frozen parent shared-tensor CPU audit plus deterministic visual-only seed42 initialization hash',retained_tensors=len(model.state_dict()),text_or_pretrained_asset_loaded=False,production_model_has_no_text_features_modules_or_legacy_condition_parameters=True,structural_initial_sha256=digest,parameters=sum(p.numel() for p in model.parameters()))
