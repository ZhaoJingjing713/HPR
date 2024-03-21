import torch
import torch.nn as nn
from torch import Tensor, nn
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

from mmdet.models.layers.transformer import MLP
from mmdet.registry import MODELS
from mmdet.utils import InstanceList,OptInstanceList
from mmdet.structures import OptSampleList, SampleList
from mmdet.structures.bbox import bbox2roi, bbox_cxcywh_to_xyxy
from mmdet.models.detectors import DDQDETR
from mmdet.models.dense_heads import DDQDETRHead
from mmdet.models.layers import SinePositionalEncoding
from mmdet.models.layers.transformer import (DeformableDetrTransformerEncoder,DDQTransformerDecoder,
                                             inverse_sigmoid, coordinate_to_encoding,)
from mmengine.model import ModuleList

from .refiner import HybridProposalRefinerLayer
try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None

@MODELS.register_module()
class HPRDDQDETRHead(DDQDETRHead):

    def loss(self,
             hidden_states: Tensor,
             references: List[Tensor],
             enc_outputs_class: Tensor,
             enc_outputs_coord: Tensor,
             batch_data_samples: SampleList,
             dn_meta: Dict[str, int],
             aux_enc_outputs_class=None,
             aux_enc_outputs_coord=None) -> dict:

        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)

        if(aux_enc_outputs_class is not None and aux_enc_outputs_coord is not None):
            aux_enc_outputs_coord = bbox_cxcywh_to_xyxy(aux_enc_outputs_coord)
            aux_enc_outputs_coord_list = []
            for img_id in range(len(aux_enc_outputs_coord)):
                det_bboxes = aux_enc_outputs_coord[img_id]
                img_shape = batch_img_metas[img_id]['img_shape']
                det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
                det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
                aux_enc_outputs_coord_list.append(det_bboxes)
            aux_enc_outputs_coord = torch.stack(aux_enc_outputs_coord_list)
            aux_loss = self.aux_loss_for_dense.loss(
                aux_enc_outputs_class.sigmoid(), aux_enc_outputs_coord,
                [item.bboxes for item in batch_gt_instances],
                [item.labels for item in batch_gt_instances], batch_img_metas)
            for k, v in aux_loss.items():
                losses[f'aux_enc_{k}'] = v

        return losses
    
    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:

        (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
         all_layers_denoising_cls_scores, all_layers_denoising_bbox_preds) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_bbox_preds, dn_meta)

        num_dense_queries = dn_meta['num_dense_queries'] if dn_meta is not None else 0
        num_layer = all_layers_matching_bbox_preds.size(0)
        dense_all_layers_matching_cls_scores = all_layers_matching_cls_scores[:, :,  # noqa: E501
                                                                              -num_dense_queries:]  # noqa: E501
        dense_all_layers_matching_bbox_preds = all_layers_matching_bbox_preds[:, :,  # noqa: E501
                                                                              -num_dense_queries:]  # noqa: E501

        all_layers_matching_cls_scores = all_layers_matching_cls_scores[:, :, :  # noqa: E501
                                                                        -num_dense_queries]  # noqa: E501
        all_layers_matching_bbox_preds = all_layers_matching_bbox_preds[:, :, :  # noqa: E501
                                                                        -num_dense_queries]  # noqa: E501

        loss_dict = self.loss_for_distinct_queries(
            all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
            batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)

        if enc_cls_scores is not None:

            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        if all_layers_denoising_cls_scores is not None:
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1],
                                  dn_losses_iou[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i

        for l_id in range(num_layer):
            cls_scores = dense_all_layers_matching_cls_scores[l_id].sigmoid()
            bbox_preds = dense_all_layers_matching_bbox_preds[l_id]

            bbox_preds = bbox_cxcywh_to_xyxy(bbox_preds)
            bbox_preds_list = []
            for img_id in range(len(bbox_preds)):
                det_bboxes = bbox_preds[img_id]
                img_shape = batch_img_metas[img_id]['img_shape']
                det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
                det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
                bbox_preds_list.append(det_bboxes)
            bbox_preds = torch.stack(bbox_preds_list)
            aux_loss = self.aux_loss_for_dense.loss(
                cls_scores, bbox_preds,
                [item.bboxes for item in batch_gt_instances],
                [item.labels for item in batch_gt_instances], batch_img_metas)
            for k, v in aux_loss.items():
                loss_dict[f'{l_id}_aux_{k}'] = v

        return loss_dict
    
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

@MODELS.register_module()
class HPRDDQDETR(DDQDETR):

    def __init__(self, *args, aux_weights=[0.5,0.5], backbone_cp=False, neck_cp=False, **kwargs):
        self.aux_weights=aux_weights
        self.backbone_cp=backbone_cp
        self.neck_cp=neck_cp
        super().__init__(*args, **kwargs)
        if(self.backbone_cp):
            checkpoint_wrapper(self.backbone)
        if(self.neck_cp):
            checkpoint_wrapper(self.neck)


    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        
        self.decoder =HPRDDQTransformerDecoder(**self.decoder_cfg)
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
        self.query_embedding = None
        self.query_map = nn.Linear(self.embed_dims, self.embed_dims)
        self.bbox_head_aux=ModuleList([deepcopy(self.bbox_head) \
                                              for _ in range(getattr(self.decoder.layers[0], 'aux_num', 0))])
        for bbox_head_aux in self.bbox_head_aux:
            bbox_head_aux.reg_branches=bbox_head_aux.reg_branches[:-2]
            bbox_head_aux.cls_branches=bbox_head_aux.cls_branches[:-2]

    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        **kwargs) -> Dict:

        inter_states, references, aux_inter_states, aux_references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
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
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        mlvl_mem=[]
        memory=encoder_outputs_dict['memory']
        level_start_index=encoder_inputs_dict['level_start_index']
        spatial_shapes=encoder_inputs_dict['spatial_shapes']

        bs,num_values,dim=memory.shape
        level_start_index_all=torch.cat([level_start_index,memory.new_tensor([num_values])])
        for idx, feat_size in enumerate(spatial_shapes):
            start_index=level_start_index_all[idx].int()
            end_index=level_start_index_all[idx+1].int()
            level_feat=memory[:,start_index:end_index,:]
            level_feat=level_feat.permute(0,2,1).contiguous()
            level_feat=level_feat.view(bs,dim, feat_size[0],feat_size[1])
            mlvl_mem.append(level_feat)

        imgs_whwh = []
        for batch_data_samples_i in batch_data_samples:
            h, w = batch_data_samples_i.metainfo['img_shape']
            imgs_whwh.append(memory.new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict, mlvl_mem=mlvl_mem, imgs_whwh=imgs_whwh)
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
      
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
  
        aux_head_inputs_dict=dict()
        aux_head_inputs_dict['hidden_states']=head_inputs_dict.pop('aux_hidden_states',None)
        aux_head_inputs_dict['references']=head_inputs_dict.pop('aux_references',None)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        if(aux_head_inputs_dict['hidden_states'] is not None and 
           aux_head_inputs_dict['references'] is not None ):
            num_lvl,bs,num_query,dim=head_inputs_dict['hidden_states'].shape
            aux_hidden_states=aux_head_inputs_dict['hidden_states'].split(num_query, dim=2)
            aux_references=aux_head_inputs_dict['references'].split(num_query, dim=2)
            aux_losses=dict()
            for aux_num, (aux_hedden_states_i, aux_references_i, aux_weight) in \
                enumerate(zip(aux_hidden_states,aux_references, self.aux_weights)):
                aux_losses_i=self.bbox_head_aux[aux_num].loss(
                    hidden_states=aux_hedden_states_i,
                    references=aux_references_i,
                    enc_outputs_class=None,
                    enc_outputs_coord=None,
                    dn_meta=head_inputs_dict['dn_meta'],
                    batch_data_samples=batch_data_samples,
                )
                aux_losses_i_=dict()
                for k,v in  aux_losses_i.items():
                    if(isinstance(v,list)):
                        for loss_idx in range(len(v)):
                            aux_losses_i_[f'hpr_{aux_num}_{k}_{loss_idx}']=aux_weight*v[loss_idx]
                    else:
                        aux_losses_i_[f'hpr_{aux_num}_{k}']=aux_weight*v
                aux_losses_i=aux_losses_i_
                aux_losses.update(aux_losses_i)
            losses.update(aux_losses)
 
        return losses

class HPRDDQTransformerDecoder(DDQTransformerDecoder):
    def __init__(self, *args, bbox_roi_extractor, **kwargs) -> None:
        self.bbox_roi_extractor=bbox_roi_extractor
        super().__init__( *args, **kwargs)

    def _init_layers(self) -> None:
        self.layers = ModuleList([
            HybridProposalRefinerLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)
        
        bbox_roi_extractor=self.bbox_roi_extractor
        self.norm_aux=ModuleList([deepcopy(self.norm) for _ in range(self.layers[0].aux_num)])
        self.bbox_roi_extractor=ModuleList([MODELS.build(bbox_roi_extractor) for _ in range(self.num_layers)])

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                mlvl_mem=None, imgs_whwh=None,
                **kwargs) -> Tensor:
        
        intermediate_aux=[]
        intermediate_reference_points_aux = [reference_points.repeat(1,self.layers[0].aux_num,1)]

        intermediate = []
        intermediate_reference_points = [reference_points]
        self.cache_dict['distinct_query_mask'] = []
        if self_attn_mask is None:
            self_attn_mask = torch.zeros((query.size(1), query.size(1)),
                                         device=query.device).bool()

        self_attn_mask = self_attn_mask[None].repeat(
            len(query) * self.cache_dict['num_heads'], 1, 1)
        for layer_index, layer in enumerate(self.layers):
            reference_bboxes_aux=bbox_cxcywh_to_xyxy(reference_points)*imgs_whwh
            bbox_roi_extractor=self.bbox_roi_extractor[layer_index]
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
                reference_points_input[:, :, 0, :],
                num_feats=self.embed_dims // 2)
            query_pos = self.ref_point_head(query_sine_embed)

            query, query_aux = layer(
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

            if not self.training:
                tmp = reg_branches[layer_index](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
                if layer_index < (len(self.layers) - 1):
                    self_attn_mask = self.select_distinct_queries(
                        reference_points, query, self_attn_mask, layer_index)

            else:
                num_dense = self.cache_dict['num_dense_queries']
                tmp = reg_branches[layer_index](query[:, :-num_dense])
                tmp_dense = self.aux_reg_branches[layer_index](
                    query[:, -num_dense:])

                tmp = torch.cat([tmp, tmp_dense], dim=1)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
                if layer_index < (len(self.layers) - 1):
                    self_attn_mask = self.select_distinct_queries(
                        reference_points, query, self_attn_mask, layer_index)

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
            if(self.training):
                return torch.stack(intermediate), torch.stack(intermediate_reference_points),\
                    torch.stack(intermediate_aux), torch.stack(intermediate_reference_points_aux)
            else:
                return torch.stack(intermediate), torch.stack(intermediate_reference_points), None, None
        if(self.training):
            return query, reference_points, query_aux, reference_points.repeat(1,self.layers[0].aux_num,1)
        else:
            return query, reference_points, query_aux, None, None

