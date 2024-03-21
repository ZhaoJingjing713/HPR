import torch
import torch.nn as nn
from torch import Tensor, nn
from torch.nn.init import normal_
from typing import Dict, List, Tuple, Optional

from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.models.detectors import DINO,DeformableDETR
from mmdet.models.dense_heads import DINOHead
from mmdet.models.layers import SinePositionalEncoding
from mmdet.models.layers.transformer import DeformableDetrTransformerEncoder
from mmengine.model import ModuleList

from .refiner import HybridProposalRefiner

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None



@MODELS.register_module()
class HPRDETRHead(DINOHead):
    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor,
                      all_layers_bbox_preds: Tensor,
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:

        if dn_meta is not None:
            num_denoising_queries = dn_meta['num_denoising_queries']
            all_layers_denoising_cls_scores = \
                all_layers_cls_scores[:, :, : num_denoising_queries, :]
            all_layers_denoising_bbox_preds = \
                all_layers_bbox_preds[:, :, : num_denoising_queries, :]
            all_layers_matching_cls_scores = \
                all_layers_cls_scores[:, :, num_denoising_queries:, :]
            all_layers_matching_bbox_preds = \
                all_layers_bbox_preds[:, :, num_denoising_queries:, :]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
        return (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds)
    
    def loss(self, hidden_states: Tensor, references: List[Tensor],
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:

        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)
        
        return losses
   
@MODELS.register_module()
class HPRDETR(DINO):
    def __init__(self, 
                 *args, 
                 aux_weights,
                 use_lsj=False,
                 ckpt_backbone=False,
                 ckpt_neck=False,
                 use_dn=False, 
                 bbox_head=None,
                 **kwargs) -> None:
        self.use_lsj=use_lsj
        self.ckpt_backbone=ckpt_backbone
        self.ckpt_neck=ckpt_neck
        self.aux_weights=aux_weights
        self.use_dn=use_dn
        super(HPRDETR, self).__init__(*args,bbox_head=bbox_head, **kwargs)
        if bbox_head is not None:
            bbox_head['num_pred_layer'] = self.decoder.num_layers
        if(self.use_dn==False):
            del self.dn_query_generator
        self.bbox_head_aux=ModuleList([MODELS.build(bbox_head) \
                                              for _ in range(getattr(self.decoder.layers[0], 'aux_num', 0))])
        if(self.ckpt_backbone):
            self.backbone=checkpoint_wrapper(self.backbone)
        if(self.ckpt_neck):
            self.neck=checkpoint_wrapper(self.neck)
    
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = HybridProposalRefiner(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for coder in self.encoder, self.decoder:
            for n, p in coder.named_parameters():
                if p.dim() > 1 and 'merger' not in n:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        nn.init.xavier_uniform_(self.query_embedding.weight)
        normal_(self.level_embed)
    
    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:

        bs, _, c = memory.shape
        imgs_whwh = []
        for batch_data_samples_i in batch_data_samples:
            h, w = batch_data_samples_i.metainfo['img_shape']
            imgs_whwh.append(memory.new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]

        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](
                output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], 
            k=self.num_queries, dim=1)[1]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()  

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)

        if self.training and self.use_dn:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            valid_ratios=valid_ratios,
            memory_mask=memory_mask,
            imgs_whwh=imgs_whwh,
            dn_mask=dn_mask)
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        
        return decoder_inputs_dict, head_inputs_dict
    
    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        mlvl_mem: Tensor,
                        imgs_whwh: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        **kwargs) -> Dict:
        
        inter_states, references, aux_inter_states, aux_references= self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            mlvl_mem=mlvl_mem,
            imgs_whwh=imgs_whwh,
            reg_branches=self.bbox_head.reg_branches,
            **kwargs)

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += \
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        if(self.training):
            decoder_outputs_dict['aux_hidden_states']=aux_inter_states
            decoder_outputs_dict['aux_references']=aux_references

        return decoder_outputs_dict


    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:

        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples,
            valid_ratios=decoder_inputs_dict['valid_ratios'],
            )
        decoder_inputs_dict.update(tmp_dec_in)

        mlvl_mem=[]
        memory=decoder_inputs_dict['memory']
        level_start_index=decoder_inputs_dict['level_start_index']
        spatial_shapes=decoder_inputs_dict['spatial_shapes']

        bs,num_values,dim=memory.shape
        level_start_index_all=torch.cat([level_start_index,memory.new_tensor([num_values])])
        for idx, feat_size in enumerate(spatial_shapes):
            start_index=level_start_index_all[idx].int()
            end_index=level_start_index_all[idx+1].int()
            level_feat=memory[:,start_index:end_index,:]
            level_feat=level_feat.permute(0,2,1).contiguous()
            level_feat=level_feat.view(bs,dim, feat_size[0],feat_size[1])
            mlvl_mem.append(level_feat)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict,
                                                    mlvl_mem=mlvl_mem)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def loss(self, 
             multi_batch_inputs: Dict[str, Tensor], 
             multi_batch_data_samples: Dict[str, SampleList]):

        if(isinstance(multi_batch_inputs, Dict)):
            batch_inputs=torch.cat([multi_batch_inputs['weak'], multi_batch_inputs['strong']])
            batch_data_samples=[]
            for data_group in ['weak','strong']:
                data_sample=multi_batch_data_samples[data_group]
                batch_data_samples.extend(data_sample)
        else:
            batch_inputs=multi_batch_inputs
            batch_data_samples=multi_batch_data_samples

        if self.use_lsj:
            batch_input_shape = batch_data_samples[0].batch_input_shape
            for data_samples in batch_data_samples:
                img_metas = data_samples.metainfo
                input_img_h, input_img_w = batch_input_shape
                img_metas['img_shape'] = [input_img_h, input_img_w]

        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)

        aux_head_inputs_dict=dict(enc_outputs_class=None,
                                enc_outputs_coord=None,
                                dn_meta=head_inputs_dict['dn_meta'])
        aux_head_inputs_dict['hidden_states']=head_inputs_dict.pop('aux_hidden_states',None)
        aux_head_inputs_dict['references']=head_inputs_dict.pop('aux_references',None)
      
        losses = self.bbox_head.loss(
            **head_inputs_dict, 
            batch_data_samples=batch_data_samples)

        if(aux_head_inputs_dict['hidden_states'] is not None and 
           aux_head_inputs_dict['references'] is not None ):
            num_lvl,bs,num_query,dim=head_inputs_dict['hidden_states'].shape
            aux_hidden_states=aux_head_inputs_dict['hidden_states'].split(num_query, dim=2)
            aux_references=aux_head_inputs_dict['references'].split(num_query, dim=2)
            aux_losses=dict()
            for aux_num, (aux_hedden_states_i, aux_references_i, aux_weight) in enumerate(zip(aux_hidden_states,aux_references, self.aux_weights)):
                aux_losses_i=self.bbox_head_aux[aux_num].loss(
                    hidden_states=aux_hedden_states_i,
                    references=aux_references_i,
                    enc_outputs_class=None,
                    enc_outputs_coord=None,
                    dn_meta=head_inputs_dict['dn_meta'],
                    batch_data_samples=batch_data_samples,
                )
                aux_losses_i={f'hpr_{aux_num}_{k}': aux_weight*v for k,v in aux_losses_i.items()}
                aux_losses.update(aux_losses_i)
 
            losses.update(aux_losses)

        return losses
    
    
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
    
        if self.use_lsj:
            batch_input_shape = batch_data_samples[0].batch_input_shape
            for data_samples in batch_data_samples:
                img_metas = data_samples.metainfo
                input_img_h, input_img_w = batch_input_shape
                img_metas['img_shape'] = [input_img_h, input_img_w]
        return super().predict(batch_inputs, batch_data_samples, rescale)