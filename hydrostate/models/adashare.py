"""AdaShare block policies on a dilated ResNet-18, using the five-slot retrieval input."""
import itertools

import torch
from mmengine.logging import MessageHub
from torch import nn
from torch.nn import functional as F

from hydrostate.data.retrieval.contract import INPUT_GROUPS, QUALITY_CHANNELS, SENSORS, TIME_STEPS
from hydrostate.registry import MODELS
from .grouped_experts import GroupedExpertHydroStateModel


class PolicyBlock(nn.Module):
    def __init__(self, previous, width, stride, dilation, project):
        super().__init__()
        self.shortcut = nn.Sequential(nn.Conv2d(previous,width,1,stride,bias=False),
                                      nn.BatchNorm2d(width)) if project else nn.Identity()
        self.branch = nn.Sequential(nn.Conv2d(previous,width,3,stride,dilation,dilation=dilation,bias=False),
                                    nn.BatchNorm2d(width), nn.ReLU(),
                                    nn.Conv2d(width,width,3,padding=dilation,dilation=dilation,bias=False),
                                    nn.BatchNorm2d(width))

    def forward(self, x, execute):
        skip = self.shortcut(x)
        if not self.training and not execute.item():
            return skip
        return torch.lerp(skip, F.relu(skip + self.branch(x)), execute.to(skip.dtype))


@MODELS.register_module()
class AdaShareDecoder(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = tuple(groups)
        self.out_channels = 64
        channels = TIME_STEPS * (sum(SENSORS.values()) + len(INPUT_GROUPS)*(QUALITY_CHANNELS+1))
        self.stem = nn.Sequential(nn.Conv2d(channels,64,7,2,3,bias=False),nn.BatchNorm2d(64),
                                  nn.ReLU(),nn.MaxPool2d(3,2,1,ceil_mode=True))
        self.blocks = nn.ModuleList()
        previous = 64
        for width,stride,dilation in zip((64,128,256,512),(1,2,1,1),(1,1,2,4)):
            self.blocks.append(PolicyBlock(previous,width,stride,dilation,previous!=width or dilation>1))
            self.blocks.append(PolicyBlock(width,width,1,dilation,False))
            previous = width
        self.logits = nn.Parameter(torch.zeros(len(groups),len(self.blocks),2))
        self.readouts = nn.ModuleDict({task:nn.ModuleList(nn.Conv2d(previous,64,3,padding=r,dilation=r)
                                                         for r in (6,12,18,24)) for task in groups})

    def prepare(self, packed, inputs):
        x, _ = packed
        visible = torch.stack([inputs[s+'_valid'].any(1) for s in SENSORS]).any(0)
        return self.stem(x.flatten(1,2)), visible

    def forward(self, prepared, bounds):
        stem, valid = prepared
        epoch = MessageHub.get_current_instance().get_info('epoch') or 0
        if epoch == 0:
            policy = torch.ones_like(self.logits[...,0])
        elif self.training and epoch <= 3:
            n = min(len(self.blocks), 3 * epoch)
            learned = F.gumbel_softmax(self.logits[:,-n:], tau=1/epoch, hard=True, dim=-1)[...,0]
            policy = torch.cat((torch.ones_like(self.logits[:,:-n,0]),learned),dim=1)
        else:
            policy = (self.logits.argmax(-1)==0).float()
        outputs = {}
        for task, decisions in zip(self.groups,policy):
            x = stem
            for block, execute in zip(self.blocks,decisions):
                x = block(x, execute)
            outputs[task] = sum(layer(x) for layer in self.readouts[task])
        valid = F.adaptive_max_pool2d(valid.float(), x.shape[-2:]).bool()
        penalty = None
        if self.training and self.logits.requires_grad:
            sparse = -self.logits.log_softmax(-1)[...,1].mean()
            sharing = sum((a[:,0]-b[:,0]).abs().mean() for a,b in itertools.combinations(self.logits,2))
            penalty = .01 * (sparse + sharing)
        return outputs, valid, penalty


@MODELS.register_module()
class AdaShareRetrieval(GroupedExpertHydroStateModel):
    def train_step(self, data, optim_wrapper):
        epoch = MessageHub.get_current_instance().get_info('epoch') or 0
        # Keep each accumulated optimizer update entirely in one phase.
        policy_step = 1 <= epoch <= 3 and (optim_wrapper._inner_count //
                                           optim_wrapper._accumulative_counts) % 2 == 1
        for p in self.parameters():
            p.requires_grad_(policy_step if p is self.decoder.logits else not policy_step)
        return super().train_step(data, optim_wrapper)
