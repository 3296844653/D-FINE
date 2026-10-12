"""EvidenceContrastAQS: an experimental extension of the project's AQS.

Image-local, prediction-conditioned class/slot evidence banks, support versus
alternative retrieval, and ambiguity-conditioned residual transfer. No GT,
new queries, local image sampling, background labels, or additional losses.
Confidence and distribution concentration are proxies, NOT calibrated validity.
"""
import math

import torch
from torch import nn


class EvidenceContrastAQS(nn.Module):
    def __init__(self, hidden_dim, num_classes, threshold=.5, temperature=.1,
                 residual_init=.1, dim=64, slots=4, heads=4, geometry=True,
                 quality=True, contrast=True, adaptive_receiver=True):
        super().__init__()
        for name, value in (("hidden_dim", hidden_dim), ("num_classes", num_classes),
                            ("dim", dim), ("slots", slots), ("heads", heads)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"EvidenceContrastAQS {name} must be a positive integer")
        if num_classes < 2 or dim % heads:
            raise ValueError("EvidenceContrastAQS needs >=2 classes and dim divisible by heads")
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("threshold must be strictly between zero and one")
        if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(residual_init):
            raise ValueError("temperature must be positive; residual_init must be finite")
        if any(type(flag) is not bool for flag in (geometry, quality, contrast, adaptive_receiver)):
            raise ValueError("EvidenceContrastAQS ablation flags must be boolean")
        self.hidden_dim, self.num_classes = hidden_dim, num_classes
        self.dim, self.slots, self.heads = dim, slots, heads
        self.temperature = float(temperature)
        self.geometry, self.quality = geometry, quality
        self.contrast, self.adaptive_receiver = contrast, adaptive_receiver
        self.threshold_logit = nn.Parameter(torch.tensor(math.log(threshold / (1-threshold))))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))
        # Saved schema/settings let offline export read T and the actual mechanism.
        self.register_buffer("settings", torch.tensor([1, hidden_dim, num_classes, dim,
            slots, heads, temperature, geometry, quality, contrast, adaptive_receiver], dtype=torch.float32))
        self.norm = nn.LayerNorm(hidden_dim)
        self.box_embed = nn.Sequential(nn.Linear(4,32), nn.SiLU(), nn.Linear(32,hidden_dim))
        self.slot_assign = nn.Linear(hidden_dim, num_classes * slots)
        self.bank_identity = nn.Parameter(torch.empty(num_classes * slots, hidden_dim))
        nn.init.normal_(self.bank_identity, std=.02)
        self.query_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.geometry_bias = nn.Sequential(nn.Linear(6,32), nn.SiLU(), nn.Linear(32,heads))
        self.receiver = nn.Sequential(nn.Linear(5,32), nn.SiLU(), nn.Linear(32,1))
        self.fuse = nn.Sequential(nn.Linear(hidden_dim + 3*dim, hidden_dim),
                                  nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # A buffer alone would silently overwrite metadata but not Python
        # flags/T, producing predictions different from the reported settings.
        supplied = state_dict.get(prefix+"settings")
        if supplied is not None and (supplied.shape != self.settings.shape or not torch.allclose(
                supplied.detach().float().cpu(), self.settings.detach().float().cpu(), rtol=1e-6, atol=1e-7)):
            error_msgs.append(prefix+"EvidenceContrastAQS fixed settings differ: load with the original config")
        layer = state_dict.get(prefix+"applied_layer")
        if layer is not None and hasattr(self,"applied_layer") and not torch.equal(
                layer.detach().cpu(),self.applied_layer.detach().cpu()):
            error_msgs.append(prefix+"EvidenceContrastAQS decoder layer differs from checkpoint")
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,
                                      missing_keys,unexpected_keys,error_msgs)

    @staticmethod
    def _relations(boxes, bank_boxes):
        """Relative geometry to a bank's weighted mean box, not a target box."""
        q, b = boxes.unsqueeze(2), bank_boxes.unsqueeze(1)
        qs, bs = q[...,2:].clamp_min(1e-4), b[...,2:].clamp_min(1e-4)
        shift = ((b[...,:2]-q[...,:2])/qs).clamp(-8,8)
        size = (bs/qs).log().clamp(-8,8)
        lo = torch.maximum(q[...,:2]-qs/2, b[...,:2]-bs/2)
        hi = torch.minimum(q[...,:2]+qs/2, b[...,:2]+bs/2)
        inter = (hi-lo).clamp_min(0).prod(-1,keepdim=True)
        union = qs.prod(-1,keepdim=True)+bs.prod(-1,keepdim=True)-inter
        iou = inter/union.clamp_min(1e-8)
        distance = shift.square().sum(-1,keepdim=True).clamp_min(1e-8).sqrt()
        return torch.cat((shift,size,iou,distance),-1)

    @staticmethod
    def _masked_attention(logits, valid):
        # Avoid all -inf softmax: an empty image bank returns exactly zero.
        weights = logits.masked_fill(~valid[:,None,None,:], -1e4).softmax(-1)
        weights = weights * valid[:,None,None,:]
        return weights / weights.sum(-1,keepdim=True).clamp_min(1e-8)

    def forward(self, query, logits, boxes, corners, *, return_aux=False):
        if (query.ndim != 3 or query.shape[-1] != self.hidden_dim
                or logits.shape != (*query.shape[:2],self.num_classes)
                or boxes.shape != (*query.shape[:2],4)
                or corners.ndim != 3 or corners.shape[:2] != query.shape[:2]
                or corners.shape[-1] < 8 or corners.shape[-1] % 4):
            raise ValueError("Expected aligned queries, class logits, cxcywh boxes and four bin distributions")
        if query.shape[1] == 0:
            return (query,{}) if return_aux else query
        B,Q,_ = query.shape
        # Prediction statistics are conditions, not a new bbox/class-head loss.
        with torch.autocast(device_type=query.device.type, enabled=False):
            probability = logits.detach().float().sigmoid()
            score = probability.amax(-1)
            pi = probability / probability.sum(-1,keepdim=True).clamp_min(1e-8)
            top = pi.topk(2,dim=-1).values
            margin = top[...,0]-top[...,1]
            ambiguity = 1-margin
            bins = corners.detach().float().reshape(B,Q,4,-1).softmax(-1)
            entropy = -(bins * bins.clamp_min(1e-8).log()).sum(-1).mean(-1)
            concentration = (1-entropy/math.log(bins.shape[-1])).clamp(0,1)
            soft = ((score-self.threshold_logit.sigmoid())/self.temperature).sigmoid()
            hard = (score >= self.threshold_logit.sigmoid()).to(soft.dtype)
            source_gate = hard.detach()-soft.detach()+soft
            reliability = score * (.5+.5*margin) * (.5+.5*concentration)
            if not self.quality:
                reliability = score
            detached_boxes = boxes.detach().float()
        norm = self.norm(query)
        assignment_input = norm + self.box_embed(detached_boxes) * float(self.geometry)
        assignment = self.slot_assign(assignment_input).float().reshape(B,Q,self.num_classes,self.slots).softmax(-1)
        with torch.autocast(device_type=query.device.type, enabled=False):
            weights = source_gate[:,:,None,None] * reliability[:,:,None,None] * pi[:,:,:,None] * assignment
            weights = weights.flatten(2)
            mass = weights.sum(1)
            valid = mass.detach() > 1e-8
            denominator = mass.clamp_min(1e-8).unsqueeze(-1)
            bank = weights.transpose(1,2).bmm(norm.float()) / denominator
            bank_boxes = weights.transpose(1,2).bmm(detached_boxes) / denominator
            relations = self._relations(detached_boxes, bank_boxes)
        K = self.num_classes*self.slots
        head_dim = self.dim//self.heads
        q = self.query_proj(norm).reshape(B,Q,self.heads,head_dim).transpose(1,2).float()
        k = self.key_proj(bank.to(norm.dtype)+self.bank_identity).reshape(B,K,self.heads,head_dim).transpose(1,2).float()
        v = self.value_proj(bank.to(norm.dtype)).reshape(B,K,self.heads,head_dim).transpose(1,2).float()
        geometry_bias = self.geometry_bias(relations).float().permute(0,3,1,2) * float(self.geometry)
        with torch.autocast(device_type=query.device.type, enabled=False):
            affinity = q @ k.transpose(-1,-2) / math.sqrt(head_dim) + geometry_bias
            support_prior = pi.repeat_interleave(self.slots,dim=-1)
            other_prior = ((1-pi)/(self.num_classes-1)).repeat_interleave(self.slots,dim=-1)
            support_attention = self._masked_attention(affinity+support_prior[:,None].clamp_min(1e-8).log(),valid)
            other_attention = self._masked_attention(affinity+other_prior[:,None].clamp_min(1e-8).log(),valid)
            support = (support_attention @ v).transpose(1,2).reshape(B,Q,self.dim)
            other = (other_attention @ v).transpose(1,2).reshape(B,Q,self.dim)
            difference = (support-other) * float(self.contrast)
            contrast_strength = difference.square().mean(-1).clamp_min(1e-8).sqrt() / (1+support.square().mean(-1).clamp_min(1e-8).sqrt())
            receiver_stats = torch.stack((score,margin,concentration,ambiguity,contrast_strength),-1)
            # Low score alone is NOT evidence that a query should be promoted.
            receiver_prior = source_gate + (1-source_gate)*score*ambiguity
            if not self.adaptive_receiver:
                receiver_prior = source_gate
        learned_receiver = self.receiver(receiver_stats).float().sigmoid()
        if not self.adaptive_receiver:
            learned_receiver = 1 + 0*learned_receiver  # keep ablation DDP graph covered
        receiver_gate = receiver_prior * learned_receiver.squeeze(-1) * valid.any(-1)[:,None]
        fused = torch.cat((norm, support.to(norm.dtype), difference.to(norm.dtype),
                           (support*difference).to(norm.dtype)),dim=-1)
        delta = self.fuse(fused)
        output = query + (self.residual_scale.tanh()*receiver_gate).unsqueeze(-1).to(query.dtype)*delta.to(query.dtype)
        if not return_aux:
            return output
        return output, dict(source_gate=source_gate, class_probability=pi, slot_assignment=assignment,
            bank_mass=mass, bank_boxes=bank_boxes, support_attention=support_attention,
            other_attention=other_attention, support=support, other=other,
            difference=difference, receiver_gate=receiver_gate, concentration=concentration)
