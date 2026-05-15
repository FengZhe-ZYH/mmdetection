from typing import Tuple, Union

from torch import Tensor

from mmdet.models.detectors.dino import DINO
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList


@MODELS.register_module()
class DINOEvidence(DINO):
    """DINO variant that exposes FPN features to the detection head.

    The base DINO detector only passes transformer outputs into the bbox head.
    The lesion-evidence branch needs the neck features for RoIAlign, so this
    shim keeps the transformer path unchanged and forwards ``img_feats`` as an
    extra optional head argument.
    """

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(
            img_feats, batch_data_samples)
        return self.bbox_head.loss(
            **head_inputs_dict,
            batch_data_samples=batch_data_samples,
            img_feats=img_feats)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(
            img_feats, batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples,
            img_feats=img_feats)
        return self.add_pred_to_datasample(batch_data_samples, results_list)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[Tensor, ...]:
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(
            img_feats, batch_data_samples)
        return self.bbox_head.forward(**head_inputs_dict)
