from torch import Tensor
from typing import Dict, List, Tuple

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmengine.model import ModuleList

from .hpr_detr import HPRDETR
from .align_detr import AlignDETRHead

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


@MODELS.register_module()
class HPRAlignDETRHead(AlignDETRHead):
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
class HPRAlignDETR(HPRDETR):
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
            bbox_head['all_layers_num_gt_repeat'] = bbox_head['all_layers_num_gt_repeat'][:-1]
        if(self.use_dn==False):
            del self.dn_query_generator
        self.bbox_head_aux=ModuleList([MODELS.build(bbox_head) \
                                              for _ in range(getattr(self.decoder.layers[0], 'aux_num', 0))])
        if(self.ckpt_backbone):
            self.backbone=checkpoint_wrapper(self.backbone)
        if(self.ckpt_neck):
            self.neck=checkpoint_wrapper(self.neck)
    
   