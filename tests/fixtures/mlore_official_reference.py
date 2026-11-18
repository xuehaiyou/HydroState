# Yuqi Yang, CC BY-NC 4.0. Unchanged forward/cal_task_feature method bodies.
# MLoRE commit 5817007c6b73cc4393d242d3a99c7a7e0b77e6ab
# models/transformers/MLoRE_vitB.py. Test-only executable reference.
# ruff: noqa
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from einops import rearrange
INTERPOLATE_MODE = 'bilinear'

class OfficialDecoderReference(nn.Module):
    def forward(self, x, eval=True):
            x = self.patch_embed(x)
            x = self.pos_drop(x + self.pos_embed[:, 1:]) 
    
    
            # multi-scale backbone feature
            all_tasks = self.p.TASKS.NAMES
            out_feat = {task: [] for task in all_tasks}
            last_feat = {task:0 for task in all_tasks}
            out_mask = {task: 0 for task in all_tasks}
            info = {} # pass information through the pipeline
    
            x_q_list = {task:[] for task in self.p.TASKS.NAMES}
            route_feat_1 = {task: None for task in all_tasks}
            route_feat_2 = {task: None for task in all_tasks}
            route_prob_1 = [{task:None for task in all_tasks} for i in range(self.num_layers)]
            route_prob_2 = [{task:None for task in all_tasks} for i in range(self.num_layers)]
            for idx, blk in enumerate(self.blocks):
                x, attn_weight = blk(x)
                if idx + 1 in self.select_list: 
                    # extract task-specific feature at this layer
                    il = np.sum(idx >= (np.array(self.select_list) - 1)) - 1 # [0,1,2]
                    _cur_task_fea, x_q, info, route_feat_1, route_prob_1[il] = self.cal_task_feature(x, attn_weight, il, info, route_feat_1)
                    for task in all_tasks:
                        _cur_task_fea_now, route_feat_2[task], route_prob_2[il][task] = self.MLoRE_2[il](_cur_task_fea[task], task, route_feat_2[task])
                        x_q_list[task].append(x_q[task])
                        out_feat[task].append(_cur_task_fea_now)
                
            x = self.norm(x)
    
            # extract task-specific feature at the last layer
            il=self.num_layers-1
            _cur_task_fea, x_q, info, route_feat_1, route_prob_1[il] = self.cal_task_feature(x, attn_weight, il, info, route_feat_1)
            
            for task in all_tasks:
                _cur_task_fea_now, route_feat_2[task], route_prob_2[il][task] = self.MLoRE_2[il](_cur_task_fea[task], task, route_feat_2[task])
                
                x_q_list[task].append(x_q[task])
                out_feat[task].append(_cur_task_fea_now)
    
                mask_now = self.task_mask[task](torch.cat(out_feat[task], dim=1))
                mask_now = torch.softmax(mask_now, dim=1)
                for il in range(self.num_layers):
                    last_feat[task] = last_feat[task] + mask_now[:, il].unsqueeze(1) * out_feat[task][il]
            info['route_1_prob'] = route_prob_1
            info['route_2_prob'] = route_prob_2
            for task in last_feat.keys():
                last_feat[task] = F.interpolate(last_feat[task], scale_factor=4, mode=INTERPOLATE_MODE)
            return last_feat, info

    def cal_task_feature(self, x, attn_weight, il, info, route_feat):
            ''' Calculate task feature at this layer
            '''
            combined_fea = rearrange(x, 'b (h w) c -> b c h w', h=self.resolution[0], w=self.resolution[1])
    
            combined_fea = self.fea_fuse[il](combined_fea)
    
            x_q = {task:0 for task in self.p.TASKS.NAMES}
            route_feat_out = {task:0 for task in self.p.TASKS.NAMES}
            route_prob_out = {task:0 for task in self.p.TASKS.NAMES}
            for task in self.p.TASKS.NAMES:
                x_q[task], route_feat_out[task], route_prob_out[task] = self.MLoRE_1[il](combined_fea, task, route_feat[task])
    
            combined_fea = x_q
    
            return combined_fea, x_q, info, route_feat_out, route_prob_out

