from mmengine.model import BaseModule
from mmcv.cnn.bricks.transformer import  MultiheadAttention
from mmdet.models.layers.transformer import MLP
import torch
from torch import nn, Tensor
from typing import List

class AddMerge(BaseModule):
    def __init__(self, learnable, embed_dims, 
                 channel_aware, initial_weights,
                 init_cfg=None):
        super(AddMerge,self).__init__(init_cfg=init_cfg)
        self.learnable=learnable
        self.embed_dims=embed_dims
        self.channel_aware=channel_aware
        self.initial_weights=initial_weights

        if(self.learnable):
            if(self.channel_aware):
                self.weights=nn.Parameter(self.initial_weights[None].repeat(self.embed_dims,1))
            else:
                self.weights=nn.Parameter(self.initial_weights)
        else:
            self.register_buffer('weights', self.initial_weights)
    
    def forward(self, query_list:List[Tensor]):
        query_list=torch.stack(query_list,dim=-1)
        if(self.channel_aware):
            weights=self.weights[None,None,...]
            merged_query=(query_list*weights).sum(-1)
        else:
            merged_query=(query_list*self.weights[None,None,None]).sum(-1)
        return merged_query

