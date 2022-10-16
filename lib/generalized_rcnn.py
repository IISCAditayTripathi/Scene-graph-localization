# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""
import math
from collections import OrderedDict
import torch
from torch import matmul, mean, mode, nn, Tensor, std
import warnings
from typing import Tuple, List, Dict, Optional, Union


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, rpn_rel,roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.rpn_rel = rpn_rel
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    #@torch.jit.unused
    def eager_outputs(self, losses, detections, attention_maps, image_shape):
        ## type: (Dict[str, Tensor], List[Dict[str, Tensor]]), Dict[str, Tensor] -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections, attention_maps, image_shape

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # print(images)
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
        # if True:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        orig_targets = targets


        images, targets = self.transform(images, targets=targets, rel=False)
        if not self.training:
            targets[0]['boxes_gt'] = targets[0]['boxes'].clone()
    
        # Check for degenerate boxes
        # TODO: Move this to a function
        subject = []
        object = []
        relation = []
        edges = []
        relation_unique = []
        nodes_unique = []

        scene_graph = True
        
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                subject.append(target['subject_embedding'].unsqueeze(0))
                object.append(target['object_embedding'].unsqueeze(0))
                relation.append(target['relation_embedding'].unsqueeze(0))
                if scene_graph:
                    relation_unique.append(target['relation_unique'].unsqueeze(0))
                    nodes_unique.append(target['unique_nodes'].unsqueeze(0))
                    edges.append(target['edges'])
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                        " Found invalid box {} for target at index {}."
                                        .format(degen_bb, target_idx))
        
        
        attention_maps = {}
        
        image_shape = images.tensors.shape
        
        
        features = self.backbone(images.tensors)
        
        

        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        
        proposals, proposal_losses = self.rpn(images, features, targets)
        
        proposals_rel = proposals
        if self.training:
        # if True:
            detections, detector_losses = self.roi_heads(features, proposals,proposals_rel, images.image_sizes, targets, (subject, object, relation, edges, relation_unique, nodes_unique))
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        else:
            detections = self.roi_heads(features, proposals,proposals_rel, images.image_sizes, targets, (subject, object, relation, edges, relation_unique, nodes_unique))
        
        if self.training:
        
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (losses, detections)
        else:
            if self.training:
                return self.eager_outputs(losses, detections, attention_maps, image_shape)

            else:
                return detections
