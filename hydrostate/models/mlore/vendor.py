# Yuqi Yang
# Licensed under CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
# Extracted without class-body changes from YuqiYang213/MLoRE
# commit 5817007c6b73cc4393d242d3a99c7a7e0b77e6ab, models/transformers/MLoRE.py
# ruff: noqa
import math
import torch
from torch import nn

BatchNorm2d = nn.BatchNorm2d

class LoraBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, rank=6):
        super().__init__()
        self.W = nn.Conv2d(in_channels, rank, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.M = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1)

    def init_weights(self):
        nn.init.kaiming_uniform_(self.W.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W.bias)
        nn.init.kaiming_uniform_(self.M.weight, a=math.sqrt(5))
        nn.init.zeros_(self.M.bias)
    
    def forward(self, x):
        x = self.W(x)
        x = self.M(x)
        return x


class SpatialAtt(nn.Module):
    def __init__(self, dim, dim_out, im_size, with_feat):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim_out, kernel_size=1)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(dim_out)
        self.convsp = nn.Linear(im_size, 1)
        self.ln_sp = nn.LayerNorm(dim)
        self.conv2 = nn.Conv2d(dim, dim_out, kernel_size=1)
        self.conv3 = nn.Conv2d(dim_out, dim_out, kernel_size=1)
        self.with_feat = with_feat
        if with_feat:
            self.feat_linear = nn.Conv2d(dim_out *2 , dim_out *2, kernel_size=1)
    
    def forward(self, x, route_feat=None):
        n, _, h, w = x.shape
        feat = self.conv1(x)
        feat = self.ln(feat.reshape(n, -1, h * w).permute(0, 2, 1)).permute(0, 2, 1).reshape(n, -1, h, w)
        feat = self.act(feat)
        feat = self.conv3(feat)

        feat_sp = self.convsp(x.reshape(n, -1, h * w)).reshape(n, 1, -1)
        feat_sp = self.ln_sp(feat_sp).reshape(n, -1, 1, 1)
        feat_sp = self.act(feat_sp)
        feat_sp = self.conv2(feat_sp)
        
        n, c, h, w = feat.shape
        feat = torch.mean(feat.reshape(n, c, h * w), dim=2).reshape(n, c, 1, 1)
        feat = torch.cat([feat, feat_sp], dim=1)

        return feat

class MOEBlock(nn.Module):
    def __init__(self, p, final_embed_dim, im_size, kernel_size=1, with_feat=False):
        super().__init__()
        self.num_lora = len(p.rank_list)
        self.p = p
        self.lora_list_1 = nn.ModuleList()
        rank_list = p.rank_list
        for i in range(self.num_lora):
            self.lora_list_1.append(LoraBlock(final_embed_dim, final_embed_dim, kernel_size=kernel_size, rank=rank_list[i]))
            self.lora_list_1[i].init_weights()
        self.conv1 = nn.ModuleDict()
        self.conv2 = nn.ModuleDict()
        self.conv3 = nn.ModuleDict()
        self.share_conv = nn.Conv2d(final_embed_dim, final_embed_dim, kernel_size=3, padding=1)
        self.bn = nn.ModuleDict()
        self.bn_all = nn.ModuleDict()
        self.activate = nn.GELU()
        for task in self.p.TASKS.NAMES:
            self.conv1[task] = nn.Conv2d(final_embed_dim, final_embed_dim, kernel_size=1)
            self.conv3[task] = nn.Conv2d(final_embed_dim, final_embed_dim, kernel_size=1)
            self.conv2[task] = LoraBlock(final_embed_dim, final_embed_dim, kernel_size=kernel_size, rank=p.spe_rank)

            self.bn[task] = BatchNorm2d(final_embed_dim)
            self.bn_all[task] = BatchNorm2d(final_embed_dim)

        self.router_1 = nn.ModuleDict() 
        self.pre_softmax = p.pre_softmax
        self.desert_k = len(p.rank_list) - p.topk
        for task in self.p.TASKS.NAMES:
            self.router_1[task] = nn.ModuleList()
            self.router_1[task].append(SpatialAtt(final_embed_dim, final_embed_dim // 4, im_size=im_size, with_feat=with_feat))
            self.router_1[task].append(nn.Conv2d(final_embed_dim // 2, self.num_lora * 2 + 1, kernel_size=1))
        
    def forward(self, x, task, route_feat_in=None):
        out_ori = self.conv1[task](x)
        out = out_ori
        n, c, h, w = out.shape
        route_feat = self.router_1[task][0](out, route_feat_in)
        prob_all = self.router_1[task][1](route_feat).unsqueeze(2)
        prob_lora, prob_mix = prob_all[:, :self.num_lora * 2], prob_all[:, self.num_lora * 2:]
        route_1_raw, stdev_1 = prob_lora.chunk(2, dim=1)  # n, 15, 1, 1, 1
        if self.training:
            noise = torch.randn_like(route_1_raw) * stdev_1
        else:
            noise = 0
        if self.pre_softmax:
            route_1_raw = route_1_raw + noise
            route_1_indice = torch.topk(route_1_raw, self.desert_k, dim=1, largest=False)[1]
            for j in range(n):
                for i in range(self.desert_k):
                    route_1_raw[j, route_1_indice[j, i].reshape(-1)] = -1e10
            route_1 = torch.softmax(route_1_raw, dim=1)
        else:
            route_1_raw = torch.softmax(route_1_raw + noise, dim=1)
            route_1_indice = torch.topk(route_1_raw, self.desert_k, dim=1, largest=False)[1]
            route_1 = route_1_raw.clone()
            for j in range(n):
                for i in range(self.desert_k):
                    route_1[j, route_1_indice[j, i].reshape(-1)] = 0
        lora_out_1 = []
        for i in range(self.num_lora):
            lora_out_1.append(self.lora_list_1[i](out).unsqueeze(1)) # n, 1, c, h, w
        lora_out_1 = torch.cat(lora_out_1, dim=1)
        lora_out_1 = torch.sum(lora_out_1 * route_1, dim=1)
        out = self.bn_all[task](lora_out_1) + self.conv2[task](out) * prob_mix[:, 0] + self.share_conv(out.detach())
        out = self.bn[task](out)
        out = self.activate(out)

        out = self.conv3[task](out)
        return out, route_feat, route_1


from torch.nn.init import trunc_normal_

class ConvHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.mt_proj = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1), BatchNorm2d(in_channels), nn.GELU())
        trunc_normal_(self.mt_proj[0].weight, std=0.02)

        self.linear_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        nn.init.normal(self.linear_pred.bias, mean=0, std=0.02)

    def forward(self, x):
        return self.linear_pred(self.mt_proj(x))
