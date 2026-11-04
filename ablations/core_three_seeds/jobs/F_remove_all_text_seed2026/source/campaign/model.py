"""Controlled 2D condition-source experiments, isolated from the paper release."""
import hashlib
import math

import torch
from torch import nn
import torch.nn.functional as F

from tasksegment.configs import DOMAIN_NAMES
from tasksegment.models.encoder import UNetEncoder2D
from tasksegment.models.task_encoder import TaskEncoder2D
from tasksegment.models.decoder import IrisMaskDecoder2D
from tasksegment.models.dense_guidance import ConvNormAct

VARIANTS = {'A': 0, 'B': 1, 'D': 2, 'E': 4, 'F': 5, 'G': 6}


def positive_parameter(value):
    return nn.Parameter(torch.tensor(math.log(math.expm1(value)), dtype=torch.float32))


class CampaignModel(nn.Module):
    def __init__(self, features, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.encoder = UNetEncoder2D(norm_type='group', group_norm_groups=16)
        self.task_encoder = TaskEncoder2D(use_ftext=False, use_semantic_text=False,
                                        use_domain_embedding=False, use_group_embedding=False)
        # Delete unused legacy text-only parameters; forward_visual never reads these.
        for name in ('semantic_log_scale','condition_norm','condition_log_scale'):
            delattr(self.task_encoder,name)
        self.decoder = IrisMaskDecoder2D(use_ftext=False, use_dense_text_guidance=False)

    def ids(self, names, device):
        return torch.tensor([DOMAIN_NAMES.index(n) for n in names], device=device, dtype=torch.long)

    def support_tokens(self, xs, ys):
        feats = self.encoder.forward_features(xs)['bottleneck']
        return self.task_encoder.forward_visual(feats, ys).mean(0, keepdim=True)

    def segment(self, xq, visual, domain_ids):
        pyramid = self.encoder.forward_features(xq)
        visual = visual.expand(len(xq), -1, -1) if visual.shape[0] == 1 else visual
        bottleneck = pyramid['bottleneck']
        flat = bottleneck.flatten(2).transpose(1, 2)
        update, _ = self.decoder.query_to_task(flat, visual, visual, need_weights=False)
        x = self.decoder.query_norm(flat + self.decoder.dropout(update))
        x = self.decoder.bottleneck_proj(x.transpose(1, 2).reshape_as(bottleneck))
        visual_summary, _ = self.decoder._summarize_task(visual)
        probability = None
        strength = 0.
        gate = self.variant != 'G'
        x4 = self.decoder.up4(x, pyramid['skip4'], visual_summary, probability, strength * .5, gate)
        x3 = self.decoder.up3(x4, pyramid['skip3'], visual_summary, probability, strength * .25, gate)
        x2 = self.decoder.up2(x3, pyramid['skip2'], visual_summary, probability, 0., gate)
        x1 = self.decoder.up1(x2, pyramid['skip1'], visual_summary, probability, 0., gate)
        logits = self.decoder.head(x1)
        aux = [self.decoder.aux4(x4), self.decoder.aux3(x3), self.decoder.aux2(x2)] if self.training else []
        return {'pred_masks': logits, 'aux_logits': aux}

    def forward(self, xs, ys, xq, domain_ids):
        return self.segment(xq, self.support_tokens(xs, ys), domain_ids)


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
