"""Four independent AQS-based behavior refinement research candidates.

All use predicted boxes and ordinary queries, all classes, original losses.
These are task-specific designs, not claimed reproductions or established gains.
"""
import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

BEHAVIOR_REFINEMENTS = ('ClassEvidenceTransport', 'QueryContrastInitialization',
                       'GeometryAwareQueryCompetition', 'PrototypeBoundaryRefinement')


def zero_head(inputs, hidden, outputs):
    result = nn.Sequential(OrderedDict(norm=nn.LayerNorm(inputs), down=nn.Linear(inputs, hidden),
                                      act=nn.SiLU(), up=nn.Linear(hidden, outputs)))
    nn.init.zeros_(result.up.weight); nn.init.zeros_(result.up.bias)
    return result


def geometry(boxes):
    boxes = boxes.detach().float()
    low = (boxes[..., :2]-boxes[..., 2:].clamp_min(0)/2).clamp(0,1)
    high = (boxes[..., :2]+boxes[..., 2:].clamp_min(0)/2).clamp(0,1)
    return torch.cat(((low+high)/2,(high-low).clamp_min(0)),-1)


def aligned_iou(first, second):
    low1, high1 = first[..., :2]-first[..., 2:]/2, first[..., :2]+first[..., 2:]/2
    low2, high2 = second[..., :2]-second[..., 2:]/2, second[..., :2]+second[..., 2:]/2
    inter = (torch.minimum(high1,high2)-torch.maximum(low1,low2)).clamp_min(0).prod(-1)
    return inter/(first[..., 2:].prod(-1)+second[..., 2:].prod(-1)-inter).clamp_min(1e-6)


def quality_features(corners, boxes, logits):
    """Detached FDR distribution descriptors, NOT observed IoU or object validity."""
    with torch.autocast(device_type=logits.device.type, enabled=False):
        prob = corners.detach().float().reshape(*corners.shape[:2],4,-1).softmax(-1)
        entropy = -(prob*prob.clamp_min(1e-8).log()).sum(-1)/math.log(prob.shape[-1])
        maximum = prob.max(-1).values
        classes = logits.detach().float().softmax(-1)
        ambiguity = -(classes*classes.clamp_min(1e-8).log()).sum(-1,keepdim=True)/math.log(logits.shape[-1])
        size = geometry(boxes)[...,2:].clamp_min(1e-5).log()
        return torch.cat((entropy,maximum,size,logits.detach().float().sigmoid().max(-1,keepdim=True).values,ambiguity),-1)


def attend(hypotheses, tokens, valid, null_keys, null_bias):
    with torch.autocast(device_type=tokens.device.type, enabled=False):
        logits = hypotheses.float() @ tokens.float().transpose(-1,-2)/math.sqrt(tokens.shape[-1])
        logits = logits.masked_fill(~valid.unsqueeze(-2),-torch.inf)
        null = (hypotheses.float()*null_keys.float()).sum(-1)/math.sqrt(tokens.shape[-1])+null_bias.float()
        weights = torch.cat((logits,null.unsqueeze(-1)),-1).softmax(-1)
        value = weights[...,:-1] @ tokens.float()
    return value,1-weights[...,-1],weights


class BehaviorRefinementBase(nn.Module):
    uses_maps = True
    def __init__(self, hidden_dim=256, num_classes=3, dim=64, grid=3,
                 scale=.1, chunk_size=32, use_checkpoint=True):
        super().__init__()
        if (any(type(v) is not int or v < 1 for v in (hidden_dim,num_classes,dim,grid,chunk_size))
                or num_classes < 2 or dim % 8 or grid < 3 or grid % 2 == 0
                or type(use_checkpoint) is not bool or isinstance(scale,bool)
                or not math.isfinite(scale) or not 0 < scale < 1):
            raise ValueError('Invalid behavior refinement dimensions/settings')
        self.name = type(self).__name__
        self.hidden_dim,self.num_classes,self.dim,self.grid = hidden_dim,num_classes,dim,grid
        self.chunk_size,self.use_checkpoint = chunk_size,use_checkpoint
        self.enabled = True
        self.memory_enabled = self.contrast_enabled = self.geometry_enabled = self.quality_enabled = True
        self.register_buffer('settings',torch.tensor([1,BEHAVIOR_REFINEMENTS.index(self.name),
            hidden_dim,num_classes,dim,grid,round(scale*1e9),chunk_size,int(use_checkpoint)],dtype=torch.int64))
        self.raw_scale = nn.Parameter(torch.tensor(math.atanh(scale),dtype=torch.float32))
        self.query_projection = nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,dim),nn.SiLU())
        if self.uses_maps:
            def project():
                return nn.Sequential(OrderedDict(conv=nn.Conv2d(hidden_dim,dim,1,bias=False),
                    norm=nn.GroupNorm(8,dim),act=nn.SiLU()))
            self.p3_projection,self.p4_projection = project(),project()
            self.position_projection = nn.Linear(2,dim,bias=False)
            self.level_embedding = nn.Parameter(torch.randn(2,dim)*.02)
            axis = (torch.arange(grid,dtype=torch.float32)+.5)/grid-.5
            yy,xx = torch.meshgrid(axis,axis,indexing='ij')
            self.register_buffer('inner_grid',torch.stack((xx,yy),-1).flatten(0,1),persistent=False)
            self.register_buffer('ring_grid',torch.tensor([[-.75,-.75],[0,-.75],[.75,-.75],
                [-.75,0],[.75,0],[-.75,.75],[0,.75],[.75,.75]]),persistent=False)

    @property
    def scale(self): return self.raw_scale.tanh()

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        saved = state_dict.get(prefix+'settings')
        if saved is not None and (saved.shape != self.settings.shape or not torch.equal(saved.cpu(),self.settings.cpu())):
            error_msgs.append(prefix+'behavior settings differ; use the checkpoint configuration')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def prepare(self,memory,shapes):
        if not self.enabled or not self.uses_maps: return None
        h,w = shapes[0]; h4,w4 = shapes[1]
        p3 = memory[:,:h*w].transpose(1,2).reshape(memory.shape[0],self.hidden_dim,h,w)
        p4 = memory[:,h*w:h*w+h4*w4].transpose(1,2).reshape(memory.shape[0],self.hidden_dim,h4,w4)
        return self.p3_projection(p3),self.p4_projection(p4)

    def sample(self,feature,box,relative):
        b,q = box.shape[:2]
        if relative.ndim == 2: relative = relative[None,None].expand(b,q,-1,-1)
        locations = box[...,None,:2]+box[...,None,2:]*relative.float()
        valid = ((locations>=0)&(locations<=1)).all(-1)&(box[...,2:]>0).all(-1,keepdim=True)
        with torch.autocast(device_type=feature.device.type,enabled=False):
            values = F.grid_sample(feature.float(),(2*locations-1).flatten(1,2).unsqueeze(2),
                                   padding_mode='zeros',align_corners=False)
        values = values.squeeze(-1).transpose(1,2).reshape(b,q,relative.shape[-2],self.dim)
        return values.to(feature.dtype),valid

    def local(self,query,boxes,context,relative=None):
        box = geometry(boxes)
        relative = self.inner_grid if relative is None else relative
        inner,ring,inner_mask,ring_mask = [],[],[],[]
        for level,feature in enumerate(context):
            values,valid = self.sample(feature,box,relative)
            positions = self.position_projection(relative.to(feature.dtype))
            inner.append(values+positions+self.level_embedding[level]); inner_mask.append(valid)
            values,valid = self.sample(feature,box,self.ring_grid)
            ring.append(values+self.position_projection(self.ring_grid.to(feature.dtype))+self.level_embedding[level])
            ring_mask.append(valid)
        return torch.cat(inner,-2),torch.cat(ring,-2),torch.cat(inner_mask,-1),torch.cat(ring_mask,-1)

    def run_chunk(self,function,*args):
        if self.training and self.use_checkpoint and torch.is_grad_enabled():
            return checkpoint(function,*args,use_reentrant=False)
        return function(*args)

    def score_delta(self,logits,contrast,common,availability):
        contrast = contrast.float()*availability.float()
        contrast = contrast-contrast.mean(-1,keepdim=True) if self.contrast_enabled else torch.zeros_like(contrast)
        common = common.float()*availability.float().mean(-1,keepdim=True) if self.quality_enabled else torch.zeros_like(common.float())
        return logits+self.scale.to(logits.dtype)*(contrast+common).to(logits.dtype)


class ClassEvidenceTransport(BehaviorRefinementBase):
    """Class-private evidence bank transported in parallel to shared decoder queries."""
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        d,k = self.dim,self.num_classes
        self.axis_projection = nn.Linear(self.hidden_dim,d,bias=False)
        self.class_embedding = nn.Parameter(torch.randn(k,d)*.02)
        self.stage_embedding = nn.Parameter(torch.randn(3,d)*.02)
        self.null_keys = nn.Parameter(torch.randn(k,d)*.02); self.null_bias = nn.Parameter(torch.zeros(k))
        self.context_gate = nn.Linear(2*d,1)
        self.evidence_projection = nn.Sequential(nn.LayerNorm(4*d),nn.Linear(4*d,d),nn.SiLU())
        self.transport = nn.GRUCell(d,d)
        self.class_head = zero_head(3*d,d,1)
        self.common_head = zero_head(2*d+12,d,1)

    def _update_chunk(self,query,boxes,bank,previous_availability,previous_boxes,p3,p4,axes,stage):
        content = self.query_projection(query)
        anchors = self.axis_projection(F.normalize(axes.detach(),dim=-1))+self.class_embedding
        old = bank if self.memory_enabled else torch.zeros_like(bank)
        hypotheses = content.unsqueeze(-2)+anchors+old+self.stage_embedding[stage]
        inner,ring,valid_inner,valid_ring = self.local(query,boxes,(p3,p4))
        positive,availability,_ = attend(hypotheses,inner,valid_inner,self.null_keys,self.null_bias)
        context,_,_ = attend(hypotheses,ring,valid_ring,self.null_keys,self.null_bias)
        if not self.contrast_enabled: context = torch.zeros_like(context)
        gate = self.context_gate(torch.cat((positive,context),-1).to(content.dtype)).sigmoid()
        evidence = self.evidence_projection(torch.cat((content.unsqueeze(-2).expand_as(positive).float(),
            positive,context,positive-gate.float()*context),-1).to(content.dtype))
        with torch.autocast(device_type=query.device.type,enabled=False):
            update = self.transport(evidence.float().flatten(0,2),old.float().flatten(0,2)).reshape_as(old)
        stability = aligned_iou(geometry(boxes),geometry(previous_boxes)).unsqueeze(-1)
        if not self.memory_enabled: stability = torch.ones_like(stability)
        strength = (.5+.5*stability).unsqueeze(-1)*availability.unsqueeze(-1)
        next_bank = old.float()+strength*(update.float()-old.float())
        next_availability = (torch.maximum(previous_availability.float()*.8,availability)
                             if self.memory_enabled else availability)
        return next_bank.to(content.dtype),next_availability

    def update(self,query,boxes,context,state,axes,stage):
        if not self.enabled: return state
        if stage not in (0,1,2) or axes.shape != (self.num_classes,self.hidden_dim):
            raise ValueError('CET expects three ordered stages and original class axes')
        b,q,_ = query.shape
        if q == 0: return (query.new_zeros(b,0,self.num_classes,self.dim),
                           query.new_zeros(b,0,self.num_classes),boxes.detach())
        if state is None:
            bank = query.new_zeros(b,q,self.num_classes,self.dim)
            availability = query.new_zeros(b,q,self.num_classes)
            old_boxes = boxes.detach()
        else: bank,availability,old_boxes = state
        chunks = []
        for start in range(0,q,self.chunk_size):
            sl = slice(start,start+self.chunk_size)
            chunks.append(self.run_chunk(self._update_chunk,query[:,sl],boxes[:,sl],bank[:,sl],
                availability[:,sl],old_boxes[:,sl],*context,axes,stage))
        return (torch.cat([x[0] for x in chunks],1),torch.cat([x[1] for x in chunks],1),boxes.detach())

    def forward(self,query,boxes,logits,corners,context=None,state=None,axes=None):
        if not self.enabled or query.shape[1] == 0: return logits
        state = self.update(query,boxes,context,state,axes,2)
        bank,availability,_ = state
        content = self.query_projection(query)
        anchors = self.axis_projection(F.normalize(axes.detach(),dim=-1))+self.class_embedding
        contrast = self.class_head(torch.cat((bank,content.unsqueeze(-2).expand_as(bank),
                                               anchors.expand_as(bank)),-1)).squeeze(-1)
        common = self.common_head(torch.cat((content,bank.mean(-2),
            quality_features(corners,boxes,logits).to(content.dtype)),-1))
        return self.score_delta(logits,contrast,common,availability)


class QueryContrastInitialization(BehaviorRefinementBase):
    """Instance/context contrast plus latent-region relations, before Decoder L1."""
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        d = self.dim
        self.slots = nn.Parameter(torch.randn(4,d)*.02)
        self.null_keys = nn.Parameter(torch.randn(4,d)*.02); self.null_bias = nn.Parameter(torch.zeros(4))
        self.relation_query,self.relation_key = nn.Linear(d,d),nn.Linear(d,d)
        self.relation_value = nn.Linear(d,d)
        self.output = zero_head(5*d,d,self.hidden_dim)

    def _init_chunk(self,query,boxes,p3,p4):
        content = self.query_projection(query)
        hypotheses = content.unsqueeze(-2)+self.slots
        inner,ring,vi,vr = self.local(query,boxes,(p3,p4))
        foreground,availability,_ = attend(hypotheses,inner,vi,self.null_keys,self.null_bias)
        surround,_,_ = attend(hypotheses,ring,vr,self.null_keys,self.null_bias)
        contrast = foreground-surround if self.contrast_enabled else foreground
        contrast = contrast.to(content.dtype)
        relation_query,relation_key,relation_value = self.relation_query(contrast),self.relation_key(contrast),self.relation_value(contrast)
        with torch.autocast(device_type=query.device.type,enabled=False):
            scores = relation_query.float() @ relation_key.float().transpose(-1,-2)/math.sqrt(self.dim)
            relations = scores.softmax(-1) @ relation_value.float()
        delta = self.output(torch.cat((content,relations.flatten(-2).to(content.dtype)),-1))
        valid = (geometry(boxes)[...,2:]>0).all(-1,keepdim=True)
        return query+self.scale.to(query.dtype)*(delta*availability.mean(-1,keepdim=True).to(delta.dtype)*valid).to(query.dtype)

    def forward(self,query,boxes,context):
        if not self.enabled or query.shape[1] == 0: return query
        return torch.cat([self.run_chunk(self._init_chunk,query[:,s:s+self.chunk_size],
            boxes[:,s:s+self.chunk_size],*context) for s in range(0,query.shape[1],self.chunk_size)],1)


class GeometryAwareQueryCompetition(BehaviorRefinementBase):
    """Separate overlap and neighbouring-instance relation banks; no hard NMS."""
    uses_maps = False
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        d,k = self.dim,self.num_classes
        self.keys,self.values = nn.Linear(d,d),nn.Linear(d,d)
        self.geometry_projection = nn.Sequential(nn.Linear(6,d//2),nn.SiLU(),nn.Linear(d//2,2))
        self.null_bias = nn.Parameter(torch.zeros(2))
        self.class_head = zero_head(3*d+3*k+14,d,k)
        self.common_head = zero_head(3*d+3*k+14,d,1)

    def _score_chunk(self,own,boxes,logits,quality,all_content,all_boxes,all_logits,indices):
        qbox,kbox = geometry(boxes).unsqueeze(-2),geometry(all_boxes).unsqueeze(-3)
        iou = aligned_iou(qbox,kbox)
        offset = (kbox[...,:2]-qbox[...,:2])/qbox[...,2:].clamp_min(1e-4)
        ratios = (kbox[...,2:].clamp_min(1e-4)/qbox[...,2:].clamp_min(1e-4)).log().clamp(-8,8)
        distance = offset.square().sum(-1)
        geo = torch.cat((offset.clamp(-8,8),ratios,iou.unsqueeze(-1),distance.clamp_max(64).unsqueeze(-1)),-1)
        edge = self.geometry_projection(geo.to(own.dtype)).float()
        if not self.geometry_enabled: edge = torch.zeros_like(edge)
        keys,values = self.keys(all_content),self.values(all_content)
        with torch.autocast(device_type=own.device.type,enabled=False):
            appearance = own.float() @ keys.float().transpose(-1,-2)/math.sqrt(self.dim)
            same = torch.sigmoid((iou-.5)/.15)
            priors = torch.stack((same*torch.exp(-distance), (1-same)*torch.exp(-distance/4)),-2)
            if not self.geometry_enabled: priors = torch.ones_like(priors)
            scores = appearance.unsqueeze(-2)+edge.transpose(-1,-2)+priors.clamp_min(1e-8).log()
            valid = distance <= 16 if self.geometry_enabled else torch.ones_like(distance,dtype=torch.bool)
            self_mask = torch.arange(all_content.shape[1],device=own.device)[None,None,:] != indices[None,:,None]
            scores = scores.masked_fill(~(valid&self_mask).unsqueeze(-2),-torch.inf)
            null = self.null_bias.float()[None,None,:,None].expand(*scores.shape[:-1],1)
            weights = torch.cat((scores,null),-1).softmax(-1)
            contexts = weights[...,:-1] @ values.float().unsqueeze(-3)
            class_context = weights[...,:-1] @ all_logits.detach().float().sigmoid().unsqueeze(-3)
        availability = 1-weights[...,-1]
        context = contexts.flatten(-2).to(own.dtype)
        fused = torch.cat((own,context,class_context.flatten(-2).to(own.dtype),
            logits.detach().sigmoid().to(own.dtype),quality.to(own.dtype),availability.to(own.dtype)),-1)
        return self.score_delta(logits,self.class_head(fused),self.common_head(fused),
                                availability.mean(-1,keepdim=True).expand_as(logits))

    def forward(self,query,boxes,logits,corners,**kwargs):
        if not self.enabled or query.shape[1] == 0: return logits
        content = self.query_projection(query)
        quality = quality_features(corners,boxes,logits)
        if not self.quality_enabled: quality = torch.zeros_like(quality)
        results = []
        for start in range(0,query.shape[1],self.chunk_size):
            sl = slice(start,start+self.chunk_size)
            ids = torch.arange(start,min(start+self.chunk_size,query.shape[1]),device=query.device)
            results.append(self.run_chunk(self._score_chunk,content[:,sl],boxes[:,sl],logits[:,sl],quality[:,sl],
                                          content,boxes,logits,ids))
        return torch.cat(results,1)


class PrototypeBoundaryRefinement(BehaviorRefinementBase):
    """Class-anchored adaptive regions and explicit competing-class evidence."""
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        d,k = self.dim,self.num_classes
        self.axis_projection = nn.Linear(self.hidden_dim,d,bias=False)
        self.class_embedding = nn.Parameter(torch.randn(k,d)*.02)
        self.offsets = nn.Linear(d,2*self.grid**2)
        nn.init.zeros_(self.offsets.weight); nn.init.zeros_(self.offsets.bias)
        self.null_keys = nn.Parameter(torch.randn(k,d)*.02); self.null_bias = nn.Parameter(torch.zeros(k))
        self.boundary_projection = nn.Sequential(nn.LayerNorm(2*d),nn.Linear(2*d,d),nn.SiLU())
        self.class_head = zero_head(4*d+12,d,1)
        self.common_head = zero_head(2*d+12,d,1)

    def _score_chunk(self,query,boxes,logits,corners,p3,p4,axes):
        content = self.query_projection(query)
        anchors = self.axis_projection(F.normalize(axes.detach(),dim=-1))+self.class_embedding
        relative = (self.inner_grid+self.offsets(content).reshape(*content.shape[:2],-1,2).float().tanh()*.15).clamp(-.5,.5)
        inner,ring,vi,vr = self.local(query,boxes,(p3,p4),relative)
        hypotheses = content.unsqueeze(-2)+anchors
        support,availability,_ = attend(hypotheses,inner,vi,self.null_keys,self.null_bias)
        surround,_,_ = attend(hypotheses,ring,vr,self.null_keys,self.null_bias)
        evidence = (support-surround if self.contrast_enabled else support).to(content.dtype)
        left = evidence.unsqueeze(-2).expand(*evidence.shape[:-2],self.num_classes,self.num_classes,self.dim)
        right = evidence.unsqueeze(-3).expand_as(left)
        boundary = self.boundary_projection(torch.cat((left-right,left*right),-1))
        other = logits.detach().float().softmax(-1).unsqueeze(-2).expand(*logits.shape[:2],self.num_classes,self.num_classes)
        diagonal = torch.eye(self.num_classes,device=query.device,dtype=torch.bool)
        other = other.masked_fill(diagonal,0)
        other = other/other.sum(-1,keepdim=True).clamp_min(1e-6)
        competition = (other.unsqueeze(-1)*boundary.float()).sum(-2).to(content.dtype)
        quality = quality_features(corners,boxes,logits).to(content.dtype)
        if not self.quality_enabled: quality = torch.zeros_like(quality)
        fused = torch.cat((content.unsqueeze(-2).expand_as(evidence),evidence,competition,
            anchors.expand_as(evidence),quality.unsqueeze(-2).expand(*evidence.shape[:-1],12)),-1)
        contrast = self.class_head(fused).squeeze(-1)
        common = self.common_head(torch.cat((content,evidence.mean(-2),quality),-1))
        return self.score_delta(logits,contrast,common,availability)

    def forward(self,query,boxes,logits,corners,context=None,axes=None,**kwargs):
        if not self.enabled or query.shape[1] == 0: return logits
        return torch.cat([self.run_chunk(self._score_chunk,query[:,s:s+self.chunk_size],boxes[:,s:s+self.chunk_size],
            logits[:,s:s+self.chunk_size],corners[:,s:s+self.chunk_size],*context,axes)
            for s in range(0,query.shape[1],self.chunk_size)],1)


def build_behavior_refinement(name,**kwargs):
    if name not in BEHAVIOR_REFINEMENTS: raise ValueError('Unknown behavior refinement module')
    return globals()[name](**kwargs)
