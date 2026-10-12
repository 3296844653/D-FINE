"""Five independent fine-grained / medium-object structural hypotheses.

These are research candidates, not established innovations or reproductions.
All residual output projections start at zero; original detection losses train
them. No GT crops, anatomical labels, class-score boosting rule, or new queries.
Related mechanisms: local/window attention (Swin), predicted reference sampling
(Deformable DETR), and multiple part descriptors (WS-DAN). See experiment notes.
"""
import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

FINEGRAIN_TRIALS = (
    "DetailOnlyCrossGate", "WindowInteractionP3", "BoxScaleContrastL2",
    "PartInteractionL2", "PairEvidenceHead",
)


def validate(channels, dim, scale):
    if type(channels) is not int or channels < 16 or type(dim) is not int or dim < 16 or dim % 8:
        raise ValueError("Finegrain channels>=16; dim>=16 must be divisible by 8")
    if isinstance(scale, bool) or not math.isfinite(scale) or not 0 < scale <= .5:
        raise ValueError("Finegrain residual_scale must be in (0,0.5]")


def grid_size(size):
    if type(size) is not int or not 3 <= size <= 9 or size % 2 == 0:
        raise ValueError("Finegrain grid/window size must be odd, in [3,9]")


def project(channels, dim):
    # Named norms obey the existing encoder/decoder optimizer regex; no change
    # to baseline parameter groups or decay settings is needed.
    return nn.Sequential(OrderedDict(conv=nn.Conv2d(channels, dim, 1, bias=False),
                                    norm=nn.GroupNorm(8, dim), act=nn.SiLU()))


def vector(channels, dim):
    return nn.Sequential(OrderedDict(norm=nn.LayerNorm(channels),linear=nn.Linear(channels,dim)))


def residual_head(inputs, dim, outputs):
    return nn.Sequential(OrderedDict(norm=nn.LayerNorm(inputs),hidden=nn.Linear(inputs,dim),
                                    act=nn.SiLU(),output=nn.Linear(dim,outputs)))


def zero(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class TrialModule(nn.Module):
    def configure(self, mode, channels, dim, scale, *specifics):
        validate(channels, dim, scale)
        self.channels, self.dim, self.scale = channels, dim, float(scale)
        # Encode FP32 settings as integer bit patterns: baseline ModelEMA
        # averages ALL floating buffers. Fixed settings must not drift under
        # repeated EMA updates or get rounded by model.half().
        self.register_buffer("settings", torch.tensor([
            1, FINEGRAIN_TRIALS.index(mode)+1, channels, dim, scale, *specifics],
            dtype=torch.float32).view(torch.int32).to(torch.int64))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        settings = state_dict.get(prefix + "settings")
        if settings is not None and not torch.equal(settings.detach().cpu(), self.settings.detach().cpu()):
            error_msgs.append(f"{prefix}finegrain checkpoint settings do not match configuration")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                    missing_keys, unexpected_keys, error_msgs)


class DetailOnlyCrossGate(TrialModule):
    """P4 controls gates only; NEVER inject its appearance into the P3 residual.

Backbone P3 supplies signed 3/5 local contrasts; final encoder P3/P4 control
which contrasts pass. Different from CrossGate's direct context concatenation
and SADF's alignment / P5 / mixed context residual.
"""
    def __init__(self, channels=256, source_channels=256, dim=64, scale=.1):
        super().__init__()
        self.configure("DetailOnlyCrossGate", channels, dim, scale, source_channels)
        if type(source_channels) is not int or source_channels < 16:
            raise ValueError("Invalid raw backbone P3 channels")
        self.source = project(source_channels, dim)
        self.fine = project(channels, dim)
        self.context = project(channels, dim)
        self.filters = nn.ModuleList([nn.Conv2d(dim, dim, k, padding=k//2, groups=dim) for k in (3,5)])
        self.gates = nn.Sequential(nn.Conv2d(3*dim, dim, 1), nn.SiLU(), nn.Conv2d(dim, 2*dim, 1))
        self.fuse = nn.Sequential(OrderedDict(conv=nn.Conv2d(3*dim,dim,1),norm=nn.GroupNorm(8,dim),act=nn.SiLU()))
        self.output = nn.Conv2d(dim, channels, 1)
        zero(self.output)

    def forward(self, source_p3, p3, p4):
        if source_p3.shape[0] != p3.shape[0] or source_p3.shape[-2:] != p3.shape[-2:]:
            raise ValueError("Raw P3 / final P3 spatial shapes differ")
        source, fine = self.source(source_p3), self.fine(p3)
        context = F.interpolate(self.context(p4), size=p3.shape[-2:], mode="bilinear", align_corners=False)
        contrasts = [layer(source-F.avg_pool2d(source,k,1,k//2,count_include_pad=False))
                     for k,layer in zip((3,5),self.filters)]
        gates = self.gates(torch.cat((fine,context,(fine-context).abs()),1)).sigmoid().chunk(2,1)
        a,b = [g*d for g,d in zip(gates,contrasts)]
        return p3 + self.scale*self.output(self.fuse(torch.cat((a,b,a*b),1)))


class WindowInteractionP3(TrialModule):
    """P3 intra-scale window relations + signed local contrast, not scale mixing.

One unshifted attention pass with local 3x3 preconditioning (NOT full Swin).
Relative spatial bias and valid-key masking handle arbitrary P3 resolutions.
"""
    def __init__(self, channels=256, dim=64, scale=.1, window=5):
        super().__init__()
        grid_size(window)
        self.configure("WindowInteractionP3",channels,dim,scale,window)
        self.window, self.heads = window, 4
        self.projection = project(channels,dim)
        self.local = nn.Conv2d(dim,dim,3,padding=1,groups=dim)
        self.qkv = nn.Conv2d(dim,3*dim,1)
        self.bias = nn.Parameter(torch.zeros((2*window-1)**2,self.heads))
        yy,xx = torch.meshgrid(torch.arange(window),torch.arange(window),indexing="ij")
        coordinates = torch.stack((yy,xx)).flatten(1)
        relative = coordinates[:,:,None]-coordinates[:,None,:]
        relative = relative + window-1
        self.register_buffer("relative_index",relative[0]*(2*window-1)+relative[1])
        self.fuse = nn.Sequential(OrderedDict(conv=nn.Conv2d(2*dim,dim,1),norm=nn.GroupNorm(8,dim),act=nn.SiLU()))
        self.output = nn.Conv2d(dim,channels,1)
        zero(self.output)

    def partition(self, feature):
        b,c,h,w = feature.shape; s=self.window
        return feature.reshape(b,c,h//s,s,w//s,s).permute(0,2,4,3,5,1).reshape(-1,s*s,c)

    def forward(self, p3):
        feature = self.projection(p3)
        b,c,h,w = feature.shape; s=self.window
        conditioned = feature+self.local(feature)
        ph,pw = (-h)%s,(-w)%s
        qkv = F.pad(self.qkv(conditioned),(0,pw,0,ph))
        windows = self.partition(qkv).reshape(-1,s*s,3,self.heads,c//self.heads).permute(2,0,3,1,4)
        q,k,v = windows.unbind(0)
        mask = F.pad(torch.ones(b,1,h,w,device=p3.device),(0,pw,0,ph))
        mask = self.partition(mask).squeeze(-1).bool()
        with torch.autocast(device_type=p3.device.type,enabled=False):
            logits = (q.float() @ k.float().transpose(-1,-2))/math.sqrt(c//self.heads)
            bias = self.bias[self.relative_index].permute(2,0,1)
            logits = logits+bias.float().unsqueeze(0)
            logits = logits.masked_fill(~mask[:,None,None,:],-1e4)
            related = logits.softmax(-1) @ v.float()
        related = related.to(feature.dtype).transpose(1,2).reshape(-1,s*s,c)
        related = related.reshape(b,(h+ph)//s,(w+pw)//s,s,s,c).permute(0,5,1,3,2,4)
        related = related.reshape(b,c,h+ph,w+pw)[:,:,:h,:w]
        contrast = feature-F.avg_pool2d(feature,3,1,1,count_include_pad=False)
        return p3+self.scale*self.output(self.fuse(torch.cat((related-feature,contrast),1)))


def relative_grid(size):
    grid_size(size)
    axis = (torch.arange(size,dtype=torch.float32)+.5)/size-.5
    yy,xx = torch.meshgrid(axis,axis,indexing="ij")
    return torch.stack((xx,yy),-1).flatten(0,1)


def sample_points(feature, boxes, relative):
    if boxes.ndim != 3 or boxes.shape[-1] != 4 or boxes.shape[0] != feature.shape[0]:
        raise ValueError("Expected feature BCHW and normalized predicted cxcywh boxes BQ4")
    with torch.autocast(device_type=feature.device.type,enabled=False):
        box = boxes.detach().float()
        positions = box[...,None,:2]+box[...,None,2:]*relative.float()
        valid = ((positions>=0)&(positions<=1)).all(-1)
        values = F.grid_sample(feature.float(),2*positions-1,mode="bilinear",
                               padding_mode="zeros",align_corners=False)
    return values.permute(0,2,3,1).to(feature.dtype),valid


def masked_weights(logits, valid):
    weights = logits.float().masked_fill(~valid,-1e4).softmax(-1)*valid.float()
    return weights/weights.sum(-1,keepdim=True).clamp_min(1e-6)


class TokenPool(nn.Module):
    def __init__(self,channels,dim):
        super().__init__()
        self.query = vector(channels,dim)
        self.key = nn.Linear(dim,dim)
        self.value = nn.Linear(dim,dim)
        self.position = nn.Linear(2,dim,bias=False)

    def forward(self,query,tokens,valid,relative):
        content = self.query(query)
        positioned = tokens+self.position(relative).to(tokens.dtype)
        logits = (self.key(positioned)*content.unsqueeze(2)).sum(-1).float()/math.sqrt(tokens.shape[-1])
        weights = masked_weights(logits,valid).to(tokens.dtype)
        local = (self.value(tokens)*weights[...,None]).sum(2)
        return content,local


class BoxScaleContrastL2(TrialModule):
    """P3 inner-box appearance versus P4 outer-ring context at decoder L2.

Not more TopK queries. Detached predicted boxes + query-conditioned bounded
locations. The context descriptor is contrasted with the object descriptor,
not uniformly added. Later L3 can change both class and localization.
"""
    def __init__(self,channels=256,dim=64,scale=.1,grid=3):
        super().__init__(); grid_size(grid)
        self.configure("BoxScaleContrastL2",channels,dim,scale,grid,1.4)
        self.p3 = project(channels,dim); self.p4=project(channels,dim)
        inner = relative_grid(grid)
        axis = torch.tensor([-.7,0.,.7]); yy,xx=torch.meshgrid(axis,axis,indexing="ij")
        ring = torch.stack((xx,yy),-1).flatten(0,1)
        ring = ring[ring.abs().sum(-1)>0]
        self.register_buffer("inner_grid",inner); self.register_buffer("ring_grid",ring)
        self.norm = nn.LayerNorm(channels)
        self.offset = nn.Linear(channels,2*(len(inner)+len(ring))); zero(self.offset)
        self.inner_pool=TokenPool(channels,dim); self.ring_pool=TokenPool(channels,dim)
        self.gate = nn.Sequential(nn.Linear(3*dim+2,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.fuse=residual_head(3*dim,dim,channels)
        zero(self.fuse[-1])

    def forward(self,query,boxes,p3,p4):
        count=len(self.inner_grid)
        offsets=.1*self.offset(self.norm(query)).float().tanh().reshape(*query.shape[:2],-1,2)
        inner=(self.inner_grid+offsets[:,:,:count]).clamp(-.49,.49)
        ring=self.ring_grid+offsets[:,:,count:]
        a,va=sample_points(self.p3(p3),boxes,inner)
        b,vb=sample_points(self.p4(p4),boxes,ring)
        content,local=self.inner_pool(query,a,va,inner)
        _,context=self.ring_pool(query,b,vb,ring)
        geometry=boxes.detach()[...,2:].to(content.dtype)
        gate=self.gate(torch.cat((content,local,context,geometry),-1)).sigmoid()
        contrasted=gate*(local-context)
        delta=self.fuse(torch.cat((content,local,contrasted),-1))
        return query+self.scale*va.any(-1,keepdim=True).to(delta.dtype)*delta


class PartInteractionL2(TrialModule):
    """Keep four spatially biased soft parts and their directed interactions.

Parts are not supervised hands/books, and may still collapse. Fixed 5x5 ROI
samples -> query-conditioned part weights -> appearance/relative-position
pair descriptors -> query residual. Different from a single pooled ROI vector.
"""
    def __init__(self,channels=256,dim=64,scale=.1,grid=5,parts=4):
        super().__init__(); grid_size(grid)
        if parts != 4:
            raise ValueError("PartInteractionL2 first experiment uses four quadrant-biased parts")
        self.configure("PartInteractionL2",channels,dim,scale,grid,parts)
        self.parts=parts; self.feature=project(channels,dim)
        self.register_buffer("grid",relative_grid(grid))
        self.register_buffer("centres",torch.tensor([[-.25,-.25],[.25,-.25],[-.25,.25],[.25,.25]]))
        self.part_query=vector(channels,parts*dim)
        self.key=nn.Linear(dim,dim); self.value=nn.Linear(dim,dim)
        self.content=vector(channels,dim)
        self.relation=nn.Sequential(nn.Linear(2*dim+2,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.fuse=residual_head(3*dim,dim,channels)
        zero(self.fuse[-1])

    def forward(self,query,boxes,p3,p4=None):
        tokens,valid=sample_points(self.feature(p3),boxes,self.grid)
        pq=self.part_query(query).reshape(*query.shape[:2],self.parts,self.dim)
        logits=(pq[:,:, :,None,:]*self.key(tokens)[:,:,None,:,:]).sum(-1).float()/math.sqrt(self.dim)
        prior=-(self.grid[None,:,:]-self.centres[:,None,:]).square().sum(-1)/(2*.25**2)
        weights=masked_weights(logits+prior,valid[:,:,None,:]).to(tokens.dtype)
        parts=(weights[...,None]*self.value(tokens)[:,:,None,:,:]).sum(-2)
        positions=(weights.float()[...,None]*self.grid).sum(-2).to(parts.dtype)
        left,right=parts[:,:,:,None,:],parts[:,:,None,:,:]
        difference=left-right; product=left*right
        displacement=positions[:,:,:,None,:]-positions[:,:,None,:,:]
        relations=self.relation(torch.cat((difference,product,displacement),-1)).mean((2,3))
        content=self.content(query)
        delta=self.fuse(torch.cat((content,parts.mean(2),relations),-1))
        return query+self.scale*valid.any(-1,keepdim=True).to(delta.dtype)*delta


class PairEvidenceHead(TrialModule):
    """Two read/write evidence pools with independent, soft-routed corrections.

Final LQE-adjusted logits -> DETACHED ambiguity/foreground routing only.
Both directions learn from original VFL; not a fixed write boost, not forced
opposite logits. Hand's direct logit and bbox computation remain unchanged.
The original classification loss can still change shared features in training.
"""
    def __init__(self,channels=256,dim=64,scale=.1,grid=5):
        super().__init__(); grid_size(grid)
        self.configure("PairEvidenceHead",channels,dim,scale,grid)
        self.feature=project(channels,dim); self.register_buffer("grid",relative_grid(grid))
        self.query=vector(channels,dim)
        self.class_evidence=nn.Parameter(torch.randn(2,dim)*.02)
        self.key=nn.Linear(dim,dim); self.value=nn.Linear(dim,dim)
        self.position=nn.Linear(2,dim,bias=False)
        self.fuse=residual_head(5*dim,dim,2)
        zero(self.fuse[-1])

    def forward(self,query,boxes,p3,base_logits):
        if base_logits.shape != (*query.shape[:2],3):
            raise ValueError("PairEvidenceHead requires 0=hand,1=read,2=write logits")
        tokens,valid=sample_points(self.feature(p3),boxes,self.grid)
        content=self.query(query)
        evidence_query=content[:,:,None,:]+self.class_evidence
        positioned=tokens+self.position(self.grid).to(tokens.dtype)
        logits=(evidence_query[:,:,:,None,:]*self.key(positioned)[:,:,None,:,:]).sum(-1).float()/math.sqrt(self.dim)
        weights=masked_weights(logits,valid[:,:,None,:]).to(tokens.dtype)
        evidence=(weights[...,None]*self.value(tokens)[:,:,None,:,:]).sum(-2)
        read,write=evidence.unbind(2)
        delta=self.fuse(torch.cat((content,read,write,read-write,read*write),-1))
        with torch.autocast(device_type=query.device.type,enabled=False):
            scores=base_logits.detach().float().sigmoid()
            foreground=(scores[...,1]+scores[...,2])/scores.sum(-1).clamp_min(1e-6)
            conditional=base_logits.detach().float()[...,1:].softmax(-1)
            ambiguity=4*conditional[...,0]*conditional[...,1]
            route=foreground*(.25+.75*ambiguity)*valid.any(-1).float()
        delta=self.scale*route[...,None].to(delta.dtype)*delta
        return torch.cat((torch.zeros_like(delta[...,:1]),delta),-1)
