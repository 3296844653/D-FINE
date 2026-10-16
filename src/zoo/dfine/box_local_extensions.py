"""Five independent additions to the unchanged BoxLocal-L2 baseline.

Predicted features/boxes only, ordinary queries only, original detection losses.
These are research candidates, not established gains or novelty claims.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .behavior_refinement import aligned_iou, geometry, quality_features


BOX_LOCAL_EXTENSIONS = (
    'DirectionalDetailBackbone', 'SemanticChannelRelation',
    'CrossScaleNeighborhoodFusion', 'PartRelationQueryInit',
    'ReliableQueryCompetition',
)


def project(source, dim):
    return nn.Sequential(nn.Conv2d(source, dim, 1, bias=False),
                         nn.GroupNorm(8, dim), nn.SiLU())


def zero_output(module):
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)
    return module


def head(source, dim, target):
    return nn.Sequential(nn.LayerNorm(source), nn.Linear(source, dim), nn.SiLU(),
                         zero_output(nn.Linear(dim, target)))


class ExtensionBase(nn.Module):
    def __init__(self, dim=32, scale=.1, chunk=32):
        super().__init__()
        if (type(dim) is not int or dim < 16 or dim % 8 or type(chunk) is not int or chunk < 1
                or isinstance(scale, bool) or not isinstance(scale, (int, float))
                or not math.isfinite(scale) or not 0 < scale <= .5):
            raise ValueError('Invalid BoxLocal extension dimensions/scale/chunk')
        self.dim, self.scale, self.chunk = dim, float(scale), chunk
        self.enabled = True
        self.register_buffer('settings', torch.tensor([
            1, BOX_LOCAL_EXTENSIONS.index(type(self).__name__), dim,
            round(scale*1e9), chunk], dtype=torch.int64))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix+'settings')
        if saved is not None and (saved.shape != self.settings.shape
                or not torch.equal(saved.cpu(), self.settings.cpu())):
            error_msgs.append(prefix+'BoxLocal extension settings differ; use the checkpoint configuration')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


class DirectionalDetailBackbone(ExtensionBase):
    """P4-conditioned horizontal/vertical/local filters at the backbone P3 output."""
    def __init__(self, fine_channels, coarse_channels, **kwargs):
        super().__init__(**kwargs)
        d = self.dim
        self.fine, self.semantic = project(fine_channels, d), project(coarse_channels, d)
        self.filters = nn.ModuleList([
            nn.Conv2d(d, d, (1, 5), padding=(0, 2), groups=d, bias=False),
            nn.Conv2d(d, d, (5, 1), padding=(2, 0), groups=d, bias=False),
            nn.Conv2d(d, d, 3, padding=1, groups=d, bias=False),
        ])
        self.selector = nn.Conv2d(2*d, 3, 1)
        self.fuse = project(3*d, d)
        self.output = zero_output(nn.Conv2d(d, fine_channels, 1))

    def forward(self, p3, p4):
        if not self.enabled:
            return p3
        fine = self.fine(p3)
        semantic = F.interpolate(self.semantic(p4), size=p3.shape[-2:], mode='bilinear', align_corners=False)
        contrast = fine-F.avg_pool2d(fine, 3, 1, 1, count_include_pad=False)
        weights = self.selector(torch.cat((fine, semantic), 1)).float().softmax(1).to(fine.dtype)
        candidates = torch.stack([conv(contrast) for conv in self.filters], 1)
        selected = (candidates*weights.unsqueeze(2)).sum(1)
        delta = self.output(self.fuse(torch.cat((selected, contrast, semantic), 1)))
        return p3+self.scale*delta


class SemanticChannelRelation(ExtensionBase):
    """Channel relations on P5 AFTER AIFI, BEFORE the original FPN/PAN."""
    def __init__(self, channels=256, **kwargs):
        super().__init__(**kwargs)
        d = self.dim
        self.heads = 4
        self.projection = project(channels, d)
        self.qkv = nn.Conv2d(d, 3*d, 1, bias=False)
        self.log_temperature = nn.Parameter(torch.zeros(self.heads, 1, 1))
        self.fuse = project(2*d, d)
        self.output = zero_output(nn.Conv2d(d, channels, 1))

    def forward(self, p5):
        if not self.enabled:
            return p5
        feature = self.projection(p5)
        b, d, h, w = feature.shape
        q, k, v = self.qkv(feature).reshape(b, 3, self.heads, d//self.heads, h*w).unbind(1)
        with torch.autocast(device_type=p5.device.type, enabled=False):
            q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
            temperature = self.log_temperature.float().exp().clamp(.1, 10)
            weights = ((q @ k.transpose(-1, -2))*temperature).softmax(-1)
            related = (weights @ v.float()).reshape(b, d, h, w).to(feature.dtype)
        delta = self.output(self.fuse(torch.cat((feature, related-feature), 1)))
        return p5+self.scale*delta


class CrossScaleNeighborhoodFusion(ExtensionBase):
    """Each final P3 position reads nine P4 neighbours or a zero null candidate."""
    def __init__(self, channels=256, **kwargs):
        super().__init__(**kwargs)
        d = self.dim
        self.fine, self.coarse = project(channels, d), project(channels, d)
        self.query, self.key, self.value = [nn.Conv2d(d, d, 1, bias=False) for _ in range(3)]
        self.position_bias = nn.Parameter(torch.zeros(9))
        self.null_bias = nn.Parameter(torch.zeros(()))
        self.fuse = project(3*d, d)
        self.output = zero_output(nn.Conv2d(d, channels, 1))

    def forward(self, p3, p4):
        if not self.enabled:
            return p3
        fine, coarse = self.fine(p3), self.coarse(p4)
        b, d, h, w = fine.shape
        def neighbours(x):
            unfolded = F.unfold(x, 3, padding=1).reshape(b, -1, *x.shape[-2:])
            return F.interpolate(unfolded, size=(h, w), mode='nearest').reshape(b, d, 9, h, w)
        keys, values = neighbours(self.key(coarse)), neighbours(self.value(coarse))
        mask = F.unfold(torch.ones_like(coarse[:, :1]), 3, padding=1).reshape(b, 9, *coarse.shape[-2:])
        valid = F.interpolate(mask, size=(h, w), mode='nearest').bool()
        query = self.query(fine)
        with torch.autocast(device_type=p3.device.type, enabled=False):
            logits = (query.float().unsqueeze(2)*keys.float()).sum(1)/math.sqrt(d)
            logits = logits+self.position_bias.float()[None, :, None, None]
            logits = logits.masked_fill(~valid, -torch.inf)
            null = self.null_bias.float().expand(b, 1, h, w)
            weights = torch.cat((logits, null), 1).softmax(1)
            evidence = (weights[:, :-1].unsqueeze(1)*values.float()).sum(2).to(fine.dtype)
        return p3+self.scale*self.output(self.fuse(torch.cat((fine, evidence, fine-evidence), 1)))


class PartRelationQueryInit(ExtensionBase):
    """Four latent parts and their relations, only BEFORE L1 and BEFORE DN concat."""
    def __init__(self, channels=256, **kwargs):
        super().__init__(**kwargs)
        d = self.dim
        self.feature = project(channels, d)
        self.content = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, d))
        self.part_query = nn.Linear(d, 4*d)
        self.key, self.value = nn.Linear(d, d), nn.Linear(d, d)
        self.position = nn.Linear(2, d, bias=False)
        self.null_bias = nn.Parameter(torch.zeros(4))
        self.relation = nn.Sequential(nn.Linear(2*d+2, d), nn.SiLU(), nn.Linear(d, d))
        self.output = head(3*d+4, d, channels)
        axis = (torch.arange(5, dtype=torch.float32)+.5)/5-.5
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('grid', torch.stack((xx, yy), -1).flatten(0, 1), persistent=False)
        self.register_buffer('centres', torch.tensor([[-.25, -.25], [.25, -.25], [-.25, .25], [.25, .25]]), persistent=False)
        self.register_buffer('off_diagonal', ~torch.eye(4, dtype=torch.bool), persistent=False)

    def _chunk(self, query, boxes, feature):
        with torch.autocast(device_type=query.device.type, enabled=False):
            box = boxes.detach().float()
            xy = box[..., None, :2]+box[..., None, 2:]*self.grid
            valid = ((xy >= 0)&(xy <= 1)).all(-1)&(box[..., 2:] > 0).all(-1, keepdim=True)
            tokens = F.grid_sample(feature.float(), 2*xy-1, mode='bilinear', align_corners=False)
            tokens = tokens.permute(0, 2, 3, 1).to(query.dtype)
        content = self.content(query)
        positioned = tokens+self.position(self.grid.to(tokens.dtype))
        hypotheses = self.part_query(content).reshape(*query.shape[:2], 4, self.dim)
        keys, values = self.key(positioned), self.value(tokens)
        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = hypotheses.float() @ keys.float().transpose(-1, -2)/math.sqrt(self.dim)
            prior = -(self.grid[None]-self.centres[:, None]).square().sum(-1)/(2*.25**2)
            logits = (logits+prior).masked_fill(~valid.unsqueeze(-2), -torch.inf)
            null = self.null_bias.float()[None, None, :, None].expand(*logits.shape[:-1], 1)
            weights = torch.cat((logits, null), -1).softmax(-1)[..., :-1]
            availability = weights.sum(-1)
            parts = weights @ values.float()
            locations = (weights @ self.grid)/availability.clamp_min(1e-6).unsqueeze(-1)
        left, right = parts.unsqueeze(-2), parts.unsqueeze(-3)
        descriptors = torch.cat(((left-right).expand(-1, -1, 4, 4, -1),
                                  left*right, locations.unsqueeze(-2)-locations.unsqueeze(-3)), -1)
        relations = self.relation(descriptors.to(content.dtype)).float()
        strength = availability.unsqueeze(-1)*availability.unsqueeze(-2)*self.off_diagonal
        related = (relations*strength.unsqueeze(-1)).sum((-3, -2))/strength.sum((-2, -1)).clamp_min(1).unsqueeze(-1)
        fused = torch.cat((content, parts.mean(-2).to(content.dtype), related.to(content.dtype),
                           availability.to(content.dtype)), -1)
        return query+self.scale*self.output(fused)

    def forward(self, query, boxes, p3):
        if not self.enabled or query.shape[1] == 0:
            return query
        feature = self.feature(p3)
        return torch.cat([self._chunk(query[:, s:s+self.chunk], boxes[:, s:s+self.chunk], feature)
                          for s in range(0, query.shape[1], self.chunk)], 1)


class ReliableQueryCompetition(ExtensionBase):
    """Final-score relations: reliable donors, separate common/class corrections.

The neighbour bank feeds common score calibration. Class correction only sees
the overlap bank, and uses a soft receiver ambiguity gate. No hard NMS/labels.
"""
    def __init__(self, channels=256, num_classes=3, **kwargs):
        super().__init__(**kwargs)
        d, k = self.dim, num_classes
        self.content = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, d), nn.SiLU())
        self.key, self.value = nn.Linear(d, d), nn.Linear(d, d)
        self.geometry_projection = nn.Sequential(nn.Linear(6, d), nn.SiLU(), nn.Linear(d, 2))
        self.null_bias = nn.Parameter(torch.zeros(2))
        self.class_head = head(2*d+2*k+12, d, k)
        self.common_head = head(3*d+16, d, 1)
        self.raw_class_scale = nn.Parameter(torch.tensor(math.atanh(self.scale/2)))
        self.raw_common_scale = nn.Parameter(torch.tensor(math.atanh(self.scale)))

    @staticmethod
    def reliability(logits, quality):
        confidence = logits.detach().float().sigmoid().amax(-1)
        concentration = (1-quality[..., :4].float().mean(-1)).clamp(.1, 1)
        # A proxy, not measured correctness/IoU. The nonzero floor avoids pruning.
        return .05+.95*torch.sigmoid((confidence-.5)/.1)*concentration

    def _chunk(self, own, box, logits, quality, all_content, all_boxes, all_logits,
               donor_reliability, indices):
        qbox, kbox = geometry(box).unsqueeze(-2), geometry(all_boxes).unsqueeze(-3)
        iou = aligned_iou(qbox, kbox)
        offset = (kbox[..., :2]-qbox[..., :2])/qbox[..., 2:].clamp_min(1e-4)
        ratios = (kbox[..., 2:].clamp_min(1e-4)/qbox[..., 2:].clamp_min(1e-4)).log().clamp(-8, 8)
        distance = offset.square().sum(-1)
        descriptors = torch.cat((offset.clamp(-8, 8), ratios, iou.unsqueeze(-1),
                                  distance.clamp_max(64).unsqueeze(-1)), -1)
        edge = self.geometry_projection(descriptors.to(own.dtype)).float()
        keys, values = self.key(all_content), self.value(all_content)
        with torch.autocast(device_type=own.device.type, enabled=False):
            appearance = own.float() @ keys.float().transpose(-1, -2)/math.sqrt(self.dim)
            overlap = torch.sigmoid((iou-.5)/.15)
            prior = torch.stack((overlap*torch.exp(-distance),
                                  (1-overlap)*torch.exp(-distance/4)), -2)
            scores = appearance.unsqueeze(-2)+edge.transpose(-1, -2)+prior.clamp_min(1e-8).log()
            scores = scores+donor_reliability[:, None, None, :].clamp_min(1e-8).log()
            not_self = torch.arange(all_content.shape[1], device=own.device)[None, None, :] != indices[None, :, None]
            scores = scores.masked_fill(~((distance <= 16)&not_self).unsqueeze(-2), -torch.inf)
            null = self.null_bias.float()[None, None, :, None].expand(*scores.shape[:-1], 1)
            weights = torch.cat((scores, null), -1).softmax(-1)[..., :-1]
            availability = weights.sum(-1)
            contexts = weights @ values.float().unsqueeze(-3)
            overlap_classes = weights[..., 0, :] @ all_logits.detach().float().sigmoid()
            donor_quality = (weights*donor_reliability[:, None, None, :]).sum(-1)
        own_scores = logits.detach().sigmoid()
        class_input = torch.cat((own, contexts[..., 0, :].to(own.dtype), own_scores.to(own.dtype),
                                 overlap_classes.to(own.dtype), quality.to(own.dtype)), -1)
        common_input = torch.cat((own, contexts.flatten(-2).to(own.dtype), quality.to(own.dtype),
                                  availability.to(own.dtype), donor_quality.to(own.dtype)), -1)
        contrast = self.class_head(class_input).float()
        contrast = contrast-contrast.mean(-1, keepdim=True)
        common = self.common_head(common_input).float()
        top = own_scores.float().topk(2, dim=-1).values
        ambiguity = .1+.9*(1-(top[..., 0]-top[..., 1])).clamp(0, 1)
        gate = ambiguity*self.reliability(logits, quality)*availability[..., 0]
        class_delta = self.raw_class_scale.tanh()*gate.unsqueeze(-1)*contrast
        common_delta = self.raw_common_scale.tanh()*availability.mean(-1, keepdim=True)*common
        return logits+(class_delta+common_delta).to(logits.dtype)

    def forward(self, query, boxes, logits, corners):
        if not self.enabled or query.shape[1] == 0:
            return logits
        content = self.content(query)
        quality = quality_features(corners, boxes, logits)
        reliability = self.reliability(logits, quality)
        results = []
        for start in range(0, query.shape[1], self.chunk):
            sl = slice(start, start+self.chunk)
            ids = torch.arange(start, min(start+self.chunk, query.shape[1]), device=query.device)
            args = (content[:, sl], boxes[:, sl], logits[:, sl], quality[:, sl], content,
                    boxes, logits, reliability, ids)
            value = (checkpoint(self._chunk, *args, use_reentrant=False)
                     if self.training and torch.is_grad_enabled() else self._chunk(*args))
            results.append(value)
        return torch.cat(results, 1)


def build_box_local_extension(name, *, hidden_dim, fine_channels, coarse_channels, dim=32, scale=.1, chunk=32):
    kwargs = dict(dim=dim, scale=scale, chunk=chunk)
    if name == 'DirectionalDetailBackbone':
        return DirectionalDetailBackbone(fine_channels, coarse_channels, **kwargs)
    classes = {cls.__name__: cls for cls in (SemanticChannelRelation, CrossScaleNeighborhoodFusion,
                                            PartRelationQueryInit, ReliableQueryCompetition)}
    if name not in classes:
        raise ValueError('Unknown BoxLocal extension')
    return classes[name](channels=hidden_dim, **kwargs)
