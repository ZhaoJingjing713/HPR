# Copyright (c) OpenMMLab. All rights reserved.
# HPR Data Re-augmentation for large-scale jitter
import collections
import copy
from typing import List, Sequence, Union

from mmengine.dataset import BaseDataset
from mmengine.dataset import ConcatDataset as MMENGINE_ConcatDataset
from mmengine.dataset import force_full_init

from mmdet.registry import DATASETS, TRANSFORMS
from mmdet.datasets import MultiImageMixDataset

@DATASETS.register_module()
class MultiBranchMultiImageMixDataset(MultiImageMixDataset):

    def __forward_transform(self, transform, transform_type, results):
        if self._skip_type_keys is not None and \
                transform_type in self._skip_type_keys:
            return results

        if hasattr(transform, 'get_indexes'):
            for i in range(self.max_refetch):
                # Make sure the results passed the loading pipeline
                # of the original dataset is not None.
                indexes = transform.get_indexes(self.dataset)
                if not isinstance(indexes, collections.abc.Sequence):
                    indexes = [indexes]
                mix_results = [
                    copy.deepcopy(self.dataset[index]) for index in indexes
                ]
                if None not in mix_results:
                    results['mix_results'] = mix_results
                    break
            else:
                raise RuntimeError(
                    'The loading pipeline of the original dataset'
                    ' always return None. Please check the correctness '
                    'of the dataset and its pipeline.')

        for i in range(self.max_refetch):
            # To confirm the results passed the training pipeline
            # of the wrapper is not None.
            updated_results = transform(copy.deepcopy(results))
            if updated_results is not None:
                results = updated_results
                break
        else:
            raise RuntimeError(
                'The training pipeline of the dataset wrapper'
                ' always return None.Please check the correctness '
                'of the dataset and its pipeline.')

        if 'mix_results' in results:
            results.pop('mix_results')
        return results
    
    def forward_transform(self,input_dict):  
        output_dict = {'strong': dict(), 'weak': dict()}  
        
        for sub_key, sub_values in input_dict.items():  
            output_dict['strong'][sub_key]=sub_values['strong']
            output_dict['weak'][sub_key]=sub_values['weak']  
        return output_dict  

    def inverse_transform(self,input_dict):  
        output_dict = {}  
        keys=input_dict['strong'].keys()
        for k in keys:
            tmp_dict={'strong':input_dict['strong'][k], 'weak': input_dict['weak'][k]}
            output_dict[k]=tmp_dict
        
        return output_dict  

    def __getitem__(self, idx):
        results = copy.deepcopy(self.dataset[idx])
        for (transform, transform_type) in zip(self.pipeline,
                                               self.pipeline_types):
            if(transform_type!='MultiBranch'):
                new_results=dict()
                for k, v in results.items():
                    v=self.__forward_transform(transform,transform_type,v)
                    new_results[k]=v
                results=new_results
            else:
                results=self.__forward_transform(transform,transform_type,results)
                results=self.forward_transform(results)
        results=self.inverse_transform(results)
        return results