import torch
import torch.nn as nn
from torch import Tensor, nn
  
from typing import Tuple,List
from copy import deepcopy
import warnings

from mmengine.model import ModuleList, BaseModule
from mmengine.utils import deprecated_api_warning
from mmcv.cnn import build_norm_layer, build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention, MultiScaleDeformableAttention
from mmdet.models.layers.transformer import MLP
from mmdet.models.layers import SinePositionalEncoding
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType
from mmdet.structures.bbox import bbox2roi, bbox_cxcywh_to_xyxy
from mmdet.models.layers.transformer import (DinoTransformerDecoder, DeformableDetrTransformerDecoderLayer,
                                             inverse_sigmoid, coordinate_to_encoding,)
from mmdet.models.layers.transformer import DynamicConv
from .merge import AddMerge
try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None

class HybridProposalRefiner(DinoTransformerDecoder):

    def __init__(self, 
                *args,
                num_cp=-1,
                bbox_roi_extractor: ConfigType = dict(
                    type='SingleRoIExtractor',
                    finest_scale=56,
                    roi_layer=dict(
                        type='RoIAlign', output_size=7, sampling_ratio=2),
                    out_channels=256,
                    featmap_strides=[8, 16, 32, 64]),
                **kwargs):
        self.num_cp=num_cp
        self.bbox_roi_extractor=bbox_roi_extractor
        super().__init__(*args,**kwargs)
        assert self.return_intermediate,"only support return_intermediate==True"
        
    def _init_layers(self) -> None:

        self.layers=ModuleList([
                HybridProposalRefinerLayer(**self.layer_cfg)
                for _ in range(self.num_layers)
            ])
       
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])

        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)
        self.norm_aux=ModuleList([nn.LayerNorm(self.embed_dims) for _ in range(self.layers[0].aux_num)])
        bbox_roi_extractor=self.bbox_roi_extractor
        self.bbox_roi_extractor = ModuleList([
                MODELS.build(bbox_roi_extractor) 
                for _ in range(self.num_layers)
            ])
        
           
    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                mlvl_mem: List[Tensor], imgs_whwh: Tensor,
                **kwargs) -> Tensor:
        intermediate = []
        intermediate_reference_points = [reference_points]
        intermediate_aux = []
        intermediate_reference_points_aux= [reference_points.clone().repeat(1,self.layers[0].aux_num,1)]
        
        for lid, layer in enumerate(self.layers):
            reference_points_aux=reference_points.clone()
            reference_bboxes_aux=bbox_cxcywh_to_xyxy(reference_points_aux.detach())*imgs_whwh
            bbox_roi_extractor=self.bbox_roi_extractor[lid]
            reference_bbox_aux_list = [bboxes for bboxes in reference_bboxes_aux]
            rois = bbox2roi(reference_bbox_aux_list)
            roi_feats=bbox_roi_extractor(mlvl_mem[:bbox_roi_extractor.num_inputs],
                                        rois)
            
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            query,query_aux = layer(
                query,
                roi_feats=roi_feats,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)
        
            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)
    
                if(self.training and query_aux is not None):
                    num_query=query.shape[1]
                    query_aux_tuple=query_aux.split(num_query, dim=1)
                    query_aux=torch.cat([self.norm_aux[idx](query_aux_i) \
                                                  for idx, query_aux_i in enumerate(query_aux_tuple)],dim=1)
                    intermediate_aux.append(query_aux)
                    intermediate_reference_points_aux.append(new_reference_points.clone().repeat(1,layer.aux_num,1)) 

        if self.return_intermediate:
            outputs=(torch.stack(intermediate), torch.stack(intermediate_reference_points),)
            if(self.training and len(intermediate_aux)!=0 and len(intermediate_reference_points_aux)!=0):
                outputs=outputs+(torch.stack(intermediate_aux), torch.stack(intermediate_reference_points_aux),)
            else:
                outputs=outputs+(None,None,)
        else:
            raise Exception()
 
        return outputs
    

class HybridProposalRefinerLayer(DeformableDetrTransformerDecoderLayer):

    def __init__(self, 
                 *args, 
                 dy_conv_cfg, 
                 regional_ca_cfg,
                 merge_method='learnable_channel_aware',
                 initial_weights=[1, 1, 1], 
                 merge_dropout=0., 
                 **kwargs):
        self.aux_num=2
        assert merge_method in ['add','learnable','learnable_channel_aware']
        self.merge_method=merge_method
        self.dy_conv_cfg=dy_conv_cfg
        self.regional_ca_cfg=regional_ca_cfg

        self.initial_weights=torch.tensor(initial_weights,dtype=torch.float32)
        assert self.aux_num+1==len(self.initial_weights)
        self.merge_dropout=merge_dropout
        super().__init__(*args, **kwargs)
        
        
    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)

        self.ca_self_attn = MultiheadAttention(**self.self_attn_cfg)
        regional_ca_cfg=deepcopy(self.regional_ca_cfg)
        self.regional_ca=RegionalCA(**regional_ca_cfg)
        self.ca_ffn = FFN(**self.ffn_cfg)

        self.dyconv_self_attn = MultiheadAttention(**self.self_attn_cfg)
        dy_conv_cfg=deepcopy(self.dy_conv_cfg)
        dy_conv_dropout=dy_conv_cfg.pop('dropout',0.)
        self.dy_conv_dropout=nn.Dropout(dy_conv_dropout)
        self.dy_conv=DynamicConv(**dy_conv_cfg)
        self.dyconv_ffn = FFN(**self.ffn_cfg)

        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)
        self.ca_norms=deepcopy(self.norms)
        self.dyconv_norms=deepcopy(self.norms)
        
        if(self.merge_method=='add'):
            self.merger_SA=AddMerge(False,self.embed_dims,False,torch.ones((self.aux_num+1)))
            self.merger_FFN=AddMerge(False,self.embed_dims,False,torch.ones((self.aux_num+1)))
        elif(self.merge_method=='learnable'):
            self.merger_SA=AddMerge(True,self.embed_dims,False,self.initial_weights)
            self.merger_FFN=AddMerge(True,self.embed_dims,False,self.initial_weights)
        elif(self.merge_method=='learnable_channel_aware'):
            self.merger_SA=AddMerge(True,self.embed_dims,True,self.initial_weights)
            self.merger_FFN=AddMerge(True,self.embed_dims,True,self.initial_weights)

    
    def forward(self,
                query: Tensor,
                roi_feats: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                **kwargs) -> Tensor:
        dyconv_query=query.clone()
        ca_query=query.clone()
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)

        dyconv_query = self.dyconv_self_attn(
            query=dyconv_query,
            key=dyconv_query,
            value=dyconv_query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        ca_query = self.ca_self_attn(
            query=ca_query,
            key=ca_query,
            value=ca_query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](self.merger_SA([query, dyconv_query, ca_query]))
    
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[1](query)
        
        dyconv_query = self.dyconv_norms[0](dyconv_query)
        bs,num_queries,dim=dyconv_query.shape
        dyconv_query_iic=self.dy_conv(dyconv_query.flatten(0,1), roi_feats)
        dyconv_query_iic=dyconv_query_iic.view(bs,num_queries,dim)
        dyconv_query=dyconv_query+self.dy_conv_dropout(dyconv_query_iic)
        dyconv_query = self.dyconv_norms[1](dyconv_query)
        dyconv_query = self.dyconv_ffn(dyconv_query)

        ca_query = self.ca_norms[0](ca_query)
        bs,num_queries,dim=ca_query.shape
        ca_query=self.regional_ca(ca_query.flatten(0,1), roi_feats)
        ca_query=ca_query.view(bs,num_queries,dim)
        ca_query = self.ca_norms[1](ca_query)
        ca_query = self.ca_ffn(ca_query)

        query = self.ffn(query)
        query = self.norms[2](self.merger_FFN([query, dyconv_query, ca_query]))
        ca_query = self.ca_norms[2](ca_query)
        dyconv_query = self.dyconv_norms[2](dyconv_query)

        aux_query=torch.cat([ca_query,dyconv_query],dim=1)
        return query, aux_query

class RegionalCA(BaseModule):
    def __init__(self,
                 sample_num,
                 embed_dims,
                 num_heads,
                 use_key_pos=True,
                 positional_encoding=dict(
                        num_feats=128,
                        normalize=True,
                        offset=0.0,  
                        temperature=20),
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 norm_cfg=dict(type='LN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 **kwargs):
        super().__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')
        self.sample_num=sample_num
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.use_key_pos=use_key_pos
        self.batch_first = batch_first
        if(use_key_pos):
            self.positional_encoding = SinePositionalEncoding(**positional_encoding)
        self.sample_proj=nn.Linear(
            self.embed_dims, self.embed_dims*self.sample_num)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

        num_output = self.embed_dims * sample_num

        self.fc_layer = nn.Linear(num_output, self.embed_dims)
        self.fc_norm = build_norm_layer(norm_cfg, self.embed_dims)[1]

        self.activation = build_activation_layer(act_cfg)

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='RegionalCA')
    def forward_attn(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return out
    
    def forward(self, query: Tensor, input_feature: Tensor):
        N,dim,h,w=input_feature.shape
        if(self.use_key_pos):
            mask=query.new_zeros([N,h,w])
            key_pos=self.positional_encoding(mask)
            key_pos=key_pos.flatten(2).permute(2, 0, 1)
            key_pos=key_pos.permute(1,0,2)
        else:
            key_pos=None
        input_feature = input_feature.flatten(2).permute(2, 0, 1)
        input_feature = input_feature.permute(1, 0, 2)  

        ca_query=self.sample_proj(query)
        ca_query=ca_query.view(N,self.sample_num, self.embed_dims)
        features=self.forward_attn(
            query=ca_query,
            key=input_feature,
            value=input_feature,
            key_pos=key_pos
        )
        features = features.flatten(1)
        features = self.fc_layer(features)
        features = self.fc_norm(features)
        features = self.activation(features)

        return self.dropout_layer(self.proj_drop(features))+query
