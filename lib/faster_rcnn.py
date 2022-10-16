from collections import OrderedDict

import torch
from torch import fake_quantize_per_tensor_affine, nn, squeeze, unsqueeze
import torch.nn.functional as F
from torch.nn.modules import linear
from torch.nn.modules.distance import CosineSimilarity

from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign
import itertools
from ._utils import jaccard
from torch_geometric.utils import dense_to_sparse, dropout_adj
from torch_geometric.data import Batch
from torch_geometric.data import Data
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt

from torch_sparse.tensor import SparseTensor


import torch_geometric.nn as geo_nn
from torch.nn import Linear, ReLU
from torch_geometric.nn import MetaLayer
from .gnn import EdgeModel, NodeModel

from ._utils import overwrite_eps

from torch.utils.model_zoo import load_url as load_state_dict_from_url
from .anchor_utils import AnchorGenerator
from .generalized_rcnn import GeneralizedRCNN
from .rpn import RPNHead, RegionProposalNetwork
from .roi_heads import RoIHeads

from .transform import GeneralizedRCNNTransform
from .backbone_utils import resnet_fpn_backbone, _validate_resnet_trainable_layers, resnet_backbone


__all__ = [
    "FasterRCNN", "fasterrcnn_resnet50_fpn",
]


class FasterRCNN(GeneralizedRCNN):
    """
    Implements Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values of x
          between 0 and W and values of y between 0 and H
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values of x
          between 0 and W and values of y between 0 and H
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes

    Example::

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import FasterRCNN
        >>> from torchvision.models.detection.rpn import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        >>> # FasterRCNN needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the RPN generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
        >>>                                    aspect_ratios=((0.5, 1.0, 2.0),))
        >>>
        >>> # let's define what are the feature maps that we will
        >>> # use to perform the region of interest cropping, as well as
        >>> # the size of the crop after rescaling.
        >>> # if your backbone returns a Tensor, featmap_names is expected to
        >>> # be ['0']. More generally, the backbone should return an
        >>> # OrderedDict[Tensor], and in featmap_names you can choose which
        >>> # feature maps to use.
        >>> roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        >>>                                                 output_size=7,
        >>>                                                 sampling_ratio=2)
        >>>
        >>> # put the pieces together inside a FasterRCNN model
        >>> model = FasterRCNN(backbone,
        >>>                    num_classes=2,
        >>>                    rpn_anchor_generator=anchor_generator,
        >>>                    box_roi_pool=roi_pooler)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """

    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=300,   # 1000 earlier
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=300,  #1000 earlier
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=256, box_positive_fraction=0.25, # box_batch_size_per_image=512 before
                 bbox_reg_weights=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        rpn_batch_size_per_image = 1000
        
        rpn_rel = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

            box_roi_pool_rel = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)



        if box_head is None:
            representation_size = 256
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            # box_head = TwoCNNHead(
            #     in_channels=256, representation_size=representation_size
            # )
            box_head = TwoCNNHead(
                out_channels * resolution ** 2,
                representation_size)
            box_head_rel = TwoCNNHead(
                out_channels * resolution ** 2,
                representation_size)


        # if box_predictor is None:
        if True:
            # representation_size = 1024
            representation_size = 256
            box_predictor = FastRCNNPredictorJoint(
                representation_size,
                num_classes)
        box_batch_size_per_image_rel = 1000

        roi_heads = RoIHeads(
            # Box
            box_roi_pool,box_roi_pool_rel, box_head, box_head_rel,box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image,box_batch_size_per_image_rel, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, rpn_rel, roi_heads, transform)


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class TwoCNNHead(nn.Module):
    def __init__(self, in_channels, representation_size) -> None:
        super(TwoCNNHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.bn = nn.BatchNorm1d(representation_size)

        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        # x = x.mean(-1).mean(-1)

        x = F.relu(self.fc6(x))
        # x = self.bn(x)
        x = self.fc7(x)
        # x = self.bn(x)
        # x = F.relu(x)

        return x


# Code for getting the geometric features is borrowed from  https://github.com/ruotianluo/Context-aware-ZSR

def get_chw(bbox):
    # given bbox, output center, height and width
    xmin, ymin, xmax, ymax = torch.split(bbox, 1, 1)
    # [num_fg_classes, num_boxes, 1]
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    return center_x, center_y, bbox_width, bbox_height

def get_pairwise_feat(boxA, boxB):
    # Generate 6d feature given two (batches of ) boxes 
    xA, yA, wA, hA = get_chw(boxA)
    xB, yB, wB, hB = get_chw(boxB)
    feat = torch.cat([
        (xA - xB) / wA,
        (yA - yB) / hA,
        (wA.log() - wB.log()),
        (hA.log() - hB.log()),
        (xB - xA) / wB,
        (yB - yA) / hB], 1)
    return feat

def get_proposal_feat(s_roi, o_roi):
    num_props = s_roi.shape[1]
    roisO = o_roi.unsqueeze(1).repeat(1, num_props, 1, 1).reshape(-1, 4)
    roisS = s_roi.unsqueeze(2).repeat(1,1,num_props, 1).reshape(-1, 4)
    # print(roisS.shape)
    roisP = torch.stack([
        torch.min(roisS[:, 0], roisO[:, 0]),
        torch.min(roisS[:, 1], roisO[:, 1]),
        torch.max(roisS[:, 2], roisO[:, 2]),
        torch.max(roisS[:, 3], roisO[:, 3])
    ], 1)
    feat = torch.cat([
        get_pairwise_feat(roisS, roisO),
        get_pairwise_feat(roisS, roisP),
        get_pairwise_feat(roisO, roisP)], 1)
    
    return feat

def get_proposal_feat_pair_wise(s_roi, o_roi):
    num_props = s_roi.shape[1]

    roisS = s_roi
    roisO = o_roi

    feat = get_pairwise_feat(roisS, roisO)
    
    return feat


def extract_multi_position_matrix_nd(bbox_S, bbox_O):

    bs = bbox_S.shape[0]
    num_prop = bbox_S.shape[1]

    bbox_S = bbox_S.view(-1, 4)
    bbox_O = bbox_O.view(-1,4)
    xmin_S, ymin_S, xmax_S, ymax_S = torch.split(bbox_S, 1, 1)
    # [num_fg_classes, num_boxes, 1]
    bbox_width_S = xmax_S - xmin_S + 1.
    bbox_height_S = ymax_S - ymin_S + 1.
    center_x_S = 0.5 * (xmin_S + xmax_S)
    center_y_S = 0.5 * (ymin_S + ymax_S)

    xmin_O, ymin_O, xmax_O, ymax_O = torch.split(bbox_O, 1, 1)
    # [num_fg_classes, num_boxes, 1]
    bbox_width_O = xmax_O - xmin_O + 1.
    bbox_height_O = ymax_O - ymin_O + 1.
    center_x_O = 0.5 * (xmin_O + xmax_O)
    center_y_O = 0.5 * (ymin_O + ymax_O)

    bbox_width_S =  bbox_width_S.view(bs, num_prop, -1)
    bbox_height_S =  bbox_height_S.view(bs, num_prop, -1)
    center_x_S =  center_x_S.view(bs, num_prop, -1)
    center_y_S =  center_y_S.view(bs, num_prop, -1)

    bbox_width_O =  bbox_width_O.view(bs, num_prop, -1)
    bbox_height_O =  bbox_height_O.view(bs, num_prop, -1)
    center_x_O =  center_x_O.view(bs, num_prop, -1)
    center_y_O =  center_y_O.view(bs, num_prop, -1)

    # print(center_x_S.shape)

    # [num_fg_classes, num_boxes, num_boxes]
    delta_x = center_x_S - center_x_O.permute(0,2,1)
    delta_x = delta_x / bbox_width_O.permute(0,2,1)
    # delta_x = nd.log(nd.maximum(nd.abs(delta_x), 1e-3))

    delta_y = center_y_S - center_y_O.permute(0,2,1)
    delta_y = delta_y / bbox_height_O.permute(0,2,1)
    # delta_y = nd.log(nd.maximum(nd.abs(delta_y), 1e-3))

    delta_width = bbox_width_S / bbox_width_O.permute(0,2,1)
    # delta_width = nd.log(delta_width)

    delta_height = bbox_height_S / bbox_height_O.permute(0,2,1)
    # delta_height = nd.log(delta_height)

    
    position_matrix = torch.stack([delta_x, delta_y, delta_width, delta_height], 3)
    position_matrix = position_matrix.abs().clamp(min=1e-3).log()
    return position_matrix

def extract_pairwise_multi_position_embedding_nd(position_mat, feat_dim=64, wave_length=1000):
    """ Extract multi-class position embedding
    Args:
        position_mat: [num_rois, num_rois, 4]
        feat_dim: dimension of embedding feature
        wave_length:
    Returns:
        embedding: [num_rois, num_rois, feat_dim]
    """
    feat_range = torch.arange(0, feat_dim / 8).type_as(position_mat)
    dim_mat = torch.pow(position_mat.new_full((1,), wave_length),
                          (8. / feat_dim) * feat_range)
    dim_mat = dim_mat.reshape(1, 1, 1, -1)
    position_mat = (100.0 * position_mat).unsqueeze(3)
    div_mat = position_mat / dim_mat
    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)
    # embedding, [num_rois, num_rois, 4, feat_dim/4]
    embedding = torch.cat([sin_mat, cos_mat], dim=3)
    embedding = embedding.reshape(*(embedding.shape[:2] + (feat_dim,)))
    return embedding



class cross_modal_attention(nn.Module):
    def __init__(self, inplanes, div=1.0) -> None:
        super(cross_modal_attention,self).__init__()
        bn_1d = nn.BatchNorm1d
        # conv_nd = nn.Conv2d
        self.in_channels = inplanes
        self.maxpool_2d = nn.MaxPool2d(self.in_channels)

        bn = nn.BatchNorm2d
        self.div = div
        
        
        self.theta_1 = nn.Sequential(nn.Linear(self.in_channels, self.in_channels//self.div),
                                        bn_1d(self.in_channels//self.div),
                                        nn.ReLU(),
                                        nn.Linear(self.in_channels//self.div, self.in_channels//self.div, bias=True))

        self.theta_2 = nn.Sequential(nn.Linear(self.in_channels//self.div, self.in_channels//self.div),
                                        bn_1d(self.in_channels//self.div),
                                        nn.ReLU(),
                                        nn.Linear(self.in_channels//self.div, self.in_channels//self.div, bias=True))

        self.softmax = torch.nn.Softmax(dim=-1)

        nn.init.xavier_uniform_(self.theta_1[0].weight)
        nn.init.xavier_uniform_(self.theta_2[0].weight)

    def attention_util(self,image, query, is_GM=False):
        attention_feats = torch.bmm(query.unsqueeze(1), image.permute(0,2,1))
        return attention_feats

    def forward(self, image_feats, query_vector, num_props, is_GM=False):

        img_feats = self.theta_1(image_feats.view(-1, self.in_channels))

        n_channels=num_props
        batch_size = query_vector.shape[0]
 
        query_vector = self.theta_2(query_vector.squeeze(1))
        img_feats = img_feats.view(batch_size, n_channels, -1)
        attention = self.attention_util(img_feats, query_vector, is_GM)

        attention = attention.view(batch_size, n_channels) # 64 before
        
        return attention


class Net6(torch.nn.Module):
    def __init__(self, in_channels, div=1):
        super(Net6,self).__init__()
        self.div = div
        self.in_channels = in_channels
        self.op = MetaLayer(EdgeModel(div=self.div, in_dim=self.in_channels), NodeModel(div=self.div, in_dim=self.in_channels))

    def forward(self, x, edge_index, edge_attr):
        h = self.op(x, edge_index, edge_attr)
        
        return h[0], h[1]


class Net9(torch.nn.Module):
    def __init__(self, in_channels=1024, div=1):
        super(Net9,self).__init__()
        self.in_channels = in_channels
        self.div = div
        self.op = MetaLayer(EdgeModel(div=self.div,in_dim=self.in_channels), NodeModel(div=self.div, in_dim=self.in_channels))
        self.attention = cross_modal_attention(in_channels, div = self.div)
        self.query_proj = nn.Linear(in_channels//div, in_channels//div)
        self.proj_x = nn.Linear(in_channels, in_channels//div)

        self.mlp = nn.Sequential(nn.Linear(in_channels//div, in_channels//div), nn.ReLU(), nn.Linear(in_channels//div, in_channels//div))

    def forward(self, x, edge_index, edge_attr, queries=None, num_props=None):
        dim = self.in_channels
        x = x.view(len(queries), num_props, dim)
        x_list = []
        for i in range(len(queries)):
            att = self.attention(x[i].view(1, num_props, dim).repeat(queries[i].shape[0],1,1), queries[i], num_props)
            q = self.query_proj(queries[i]).unsqueeze(1).repeat(1, num_props,1)    
            att = torch.nn.functional.softmax(att.unsqueeze(-1)*1.4, dim=0)

            
            q = att*q
            x_i = self.proj_x(x[i]) + q.sum(0)

            x_i = self.mlp(x_i)
            x_i = x_i.unsqueeze(0)

            x_list.append(x_i)
        x = torch.cat(x_list, dim=0)
        
        dim = self.in_channels//self.div

        x = x.view(-1, dim)
        
        h = self.op(x, edge_index, edge_attr)

        x = h[0]
        
        return x, h[1]



class FastRCNNPredictorPairedSortedGNNFull(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictorPairedSortedGNNFull, self).__init__()
        in_channels = 1024
        n_dim = 300
        self.div = 2
        
        self.subj_proj = nn.Linear(in_channels//self.div, in_channels//self.div)
        self.obj_proj = nn.Linear(in_channels//self.div, in_channels//self.div)
        self.node_proj = nn.Linear(n_dim, in_channels//self.div)

        self.node_proj_v2 = nn.Linear(n_dim, in_channels//self.div)

        self.cls_score = nn.Linear(2*in_channels, 1) 

        self.cls_score_o = nn.Linear(2*in_channels, 1) 
        self.image2rel_s = nn.Linear(in_channels, in_channels)
        self.image2rel_o = nn.Linear(in_channels, in_channels)
        self.finetune = True
        
        self.GCN = Net9(in_channels=1024, div=self.div)
        
        self.GCN_text = Net6(in_channels=1024, div=self.div)
        
        self.is_cosine = True

        self.cls_pair_proj_v2 = nn.Sequential(
                                    nn.Linear(2*in_channels, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 128),
                                    nn.ReLU()
        )

        self.cls_score_pair = nn.Linear(128, 1)

        self.pair_fuse_feats = nn.Sequential(
                                    nn.Linear(2*in_channels+64, in_channels//self.div),
                                    nn.ReLU(),
                                    nn.Linear(in_channels//self.div, in_channels//self.div),
                                    nn.ReLU(),
                                    nn.Linear(in_channels//self.div, in_channels//self.div),

        )

        self.cls_score_pair_geo = nn.Linear(128, 1) # uncomment for geometric feats

        self.bbox_pred = nn.Linear(in_channels, (num_classes) * 4)

        n_dim = 300 # in_channels before

        self.theta_rel = nn.Sequential(
                                    nn.Linear(n_dim, in_channels//self.div)
         )
        self.theta_rel_geo = nn.Linear(in_channels, 64) # uncomment for geometric feats
        self.cos_feats = True
        if self.cos_feats:
            geo_in = 64
        else:
            geo_in = 18
        self.geo_project_v2 = nn.Sequential(nn.Linear(geo_in, 64),   # 64 otherwise
                                            # nn.BatchNorm1d(32),
                                            nn.ReLU(),
                                            nn.Linear(64,64))
        self.proj_x = nn.Linear(1024, 1024//self.div)
        train = 256
        
        self.sel = train
        self.num_prop = self.sel
        
        self.is_gnn = False
        

    def forward(self, x, edges=None, rel_unique=None, unique_nodes=None):
        subject = x[1]
        object = x[2]
        relation = x[3]
        proposals = x[4]
        
        x = x[0]
        
        
        prop = []
        for proposal in proposals:
            prop.append(proposal.unsqueeze(0))
        prop = torch.cat(prop, dim=0)
        
        self.num_prop = self.sel
        self.num_prop = prop.shape[1]
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        n_dim = 1024 # 256 before
        try:
            bs = relation.shape[0]
        except:
            bs = len(relation)
        
        
        if self.finetune:
            graph_data = []
            nodes_gcn = []
            for i in range(bs):
                y = unique_nodes[i]
                ed = torch.cat([edges[i][:,0].unsqueeze(-1), edges[i][:,2].unsqueeze(-1)], dim=-1)
                
                ed = ed.permute(1,0)
                
                size = torch.max(ed)
                ed = torch.sparse_coo_tensor(indices=ed, values=torch.ones(ed.shape[-1]).cuda(), dtype=torch.long, size=[size+1, size+1]).to_dense()
                ed = dense_to_sparse(ed)[0]
                rel = relation[i]
                try:
                    graph_data.append(Data(x = self.node_proj(y.squeeze(0).permute(1,0)), edge_index=ed, edge_attr=self.theta_rel(rel.squeeze(0).permute(1,0))))
                except:
                    graph_data.append(Data(x = self.node_proj(y.squeeze(0).permute(1,0)), edge_index=ed, edge_attr=self.theta_rel(rel.squeeze(0))))
                nodes_gcn.append(self.node_proj_v2(y.squeeze(0).permute(1,0)))
            graph_data = Batch.from_data_list(graph_data)
            if True:
                nodes, edge_attr = self.GCN_text(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
                
                graph_data.x = nodes

            node_list = graph_data.to_data_list() 
            nodes = []
            for i, node in enumerate(node_list):
                nodes.append(node.x)
        
        x_graph = x.clone()
        
        if True:
            
            x_1_full = x.clone().view(bs, self.num_prop, n_dim)
            x_2_full = x.clone().view(bs, self.num_prop, n_dim)     
            subject_proposals_full = prop.clone() # uncomment for geometric feats
            object_proposals_full = prop.clone() # uncomment for geometric feats
        
            proposals_full = self.num_prop
            if self.cos_feats:
                geo_feats_full = extract_multi_position_matrix_nd(subject_proposals_full, object_proposals_full)

                geo_feats_list = []
                for i in range(bs):
                    a = extract_pairwise_multi_position_embedding_nd(geo_feats_full[i])
                    geo_feats_list.append(a.unsqueeze(0))

                geo_feats_full = torch.cat(geo_feats_list, dim=0)
            else:
                geo_feats_full = get_proposal_feat(subject_proposals_full, object_proposals_full)
            geo_feats_full = self.geo_project_v2(geo_feats_full)

            x_1_full = self.image2rel_s(x_1_full)
            x_2_full = self.image2rel_o(x_2_full)
            
            
            x_2_full = x_2_full.unsqueeze(1).repeat(1, proposals_full,1,1)
            x_1_full = x_1_full.unsqueeze(2).repeat(1,1,proposals_full,1)
            geo_feats_full = geo_feats_full.view(bs, proposals_full, proposals_full, -1)

            pair_score_full = torch.cat([x_1_full, x_2_full, geo_feats_full], dim=-1)
            # pair_score_full = torch.cat([x_1_full, x_2_full], dim=-1)
            pair_score_full = self.pair_fuse_feats(pair_score_full.view(bs, proposals_full*proposals_full, -1))

            pair_score_full = pair_score_full.view(bs, proposals_full, proposals_full, -1)
            edge_feats = pair_score_full.clone()
            pair_score_full_list = []
            for i in range(bs):
                rel = rel_unique[i]
                rel = self.theta_rel(rel.squeeze(0).permute(1,0)).unsqueeze(1).unsqueeze(1).repeat(1,self.num_prop, self.num_prop,1)
                # print(pair_score_full[i].shape)
                pair_score_full_i = pair_score_full[i].unsqueeze(0).repeat(rel.shape[0],1,1,1)
                pair_score_full_i = (torch.nn.functional.cosine_similarity(pair_score_full_i, rel, dim=-1) +1)/2
                test_pair_score = pair_score_full_i.clone()
                if not pair_score_full_i.shape[0] == 1:
                    pair_score_full_i, _ = torch.max(pair_score_full_i,dim=0)
                    pair_score_full_i = pair_score_full_i.unsqueeze(0)
                pair_score_full_list.append(pair_score_full_i)
            
            pair_score_full = torch.cat(pair_score_full_list, dim=0)    
            pair_score_full = pair_score_full.view(bs, proposals_full, proposals_full)
            
            
            pair_score_full[pair_score_full < 0.04] = 0 # 0.04
            

            x_graph = x_graph.view(bs, self.num_prop, n_dim)
            graph_data = []
            for i in range(bs):
                pair_score_full_i = pair_score_full[i]
                
                edge_feats_i = edge_feats[i]
                pair_score_full_i.fill_diagonal_(0)
                edge_feats_i = edge_feats_i[pair_score_full_i>0]
                edge_indices = dense_to_sparse(pair_score_full_i)[0]
                graph_data.append(Data(x=x_graph[i], edge_index=edge_indices, edge_attr=edge_feats_i))
            

            graph_data = Batch.from_data_list(graph_data)
            
            
            x_graph, edge_attr = self.GCN(graph_data.x, graph_data.edge_index, graph_data.edge_attr, nodes_gcn, self.num_prop)
            

        x_graph = x_graph.view(bs, self.num_prop, n_dim//self.div)
        nodes_score = []
        node_shapes = []
        for i in range(bs):
            x_graph_i = x_graph[i].unsqueeze(0).repeat(nodes[i].shape[0],1,1)
            nodes_i = nodes[i].unsqueeze(1).repeat(1, proposals_full,1)
            node_shapes.append(nodes[i].shape[0])
            nodes_i = (torch.nn.functional.cosine_similarity(x_graph_i, nodes_i, dim=-1) +1)/2
            nodes_score.append(nodes_i)
        nodes_score_ = torch.cat(nodes_score, dim=0)
        
        
        
        
        shapes = []
        x_subj = []
        x_obj = []
        rels = []
        x = x.view(bs, self.num_prop, n_dim)
        x_expanded = []
        
        prop_expanded = []
        
        for i, edge in enumerate(edges):
            prop_i = prop[i].unsqueeze(0).repeat(edge.shape[0],1,1)
            prop_expanded.append(prop_i)
            x_i = x[i].unsqueeze(0).repeat(edge.shape[0],1,1)
            x_expanded.append(x_i)
            shapes.append(edge.shape[0])
            subj = edge[:,0]
            relation = edge[:,1]

            rel = rel_unique[i].squeeze(0).permute(1,0)[relation,:]
            rel = self.theta_rel(rel)
            rels.append(rel)
            obj = edge[:,2]
            x_subj.append(nodes_score[i][subj,:])
            x_obj.append(nodes_score[i][obj,:])
        x_subj = torch.cat(x_subj, dim=0)
        x_obj = torch.cat(x_obj, dim=0)
        x_expanded = torch.cat(x_expanded, dim=0)
        rels = torch.cat(rels, dim=0)
        prop_expanded = torch.cat(prop_expanded,dim=0)
        bs = x_subj.shape[0]
        
        x_subj = x_subj.view(bs, self.num_prop)
        x_obj = x_obj.view(bs, self.num_prop)
        
        x_subj_test = x_subj.clone()
        x_obj_test = x_obj.clone()

        x_subj, subj_indices = torch.sort(x_subj, dim=-1, descending=True)
        x_obj, obj_indices = torch.sort(x_obj, dim=-1, descending=True)
        
        if True:
            subj_indices_u = [torch.randperm(50)[0:24] for i in range(bs)]
            obj_indices_u = [torch.randperm(50)[0:24] for i in range(bs)]
            subj_indices_u = torch.cat(subj_indices_u, dim=0)
            obj_indices_u= torch.cat(obj_indices_u, dim=0)

            subj_indices_l = [torch.randperm(180)[0:24]+50 for i in range(bs)]
            obj_indices_l = [torch.randperm(180)[0:24]+50 for i in range(bs)]
            subj_indices_l = torch.cat(subj_indices_l, dim=0)
            obj_indices_l = torch.cat(obj_indices_l, dim=0)

            subj_indices_u = subj_indices_u.view(bs, -1)
            subj_indices_l = subj_indices_l.view(bs, -1)
            obj_indices_u = obj_indices_u.view(bs, -1)
            obj_indices_l = obj_indices_l.view(bs, -1)

            subj_indices = torch.cat([subj_indices_u, subj_indices_l], dim=1)
            obj_indices = torch.cat([obj_indices_u, obj_indices_l], dim=1)
        
        
        subj_indices = subj_indices.view(bs, -1)
        obj_indices = obj_indices.view(bs, -1)
        
        x_1 = x_expanded.view(bs, self.num_prop, n_dim)
        x_2 = x_expanded.view(bs, self.num_prop, n_dim)


        subject_proposals = prop_expanded.clone() # uncomment for geometric feats
        object_proposals = prop_expanded.clone() # uncomment for geometric feats

            
        if self.training:
            
            subject_new = []
            object_new = []
            for i in range(bs):
                subject_new.append(x_subj[i, subj_indices[i]].unsqueeze(0))
                object_new.append(x_obj[i, obj_indices[i]].unsqueeze(0))

            x_subj = torch.cat(subject_new, dim=0)
            x_obj = torch.cat(object_new, dim=0)
            num_samp = 48
            x_subj= x_subj[:,0:num_samp]
            
            x_obj = x_obj[:,0:num_samp]
            

            subj_indices = subj_indices[:,0:num_samp]
            
            obj_indices = obj_indices[:, 0:num_samp]
            
            orig_num_prop = self.num_prop
            self.num_prop = num_samp
        else:
            orig_num_prop = self.num_prop

        bs = x_1.shape[0]
        if self.training:
            x_1_new = []
            x_2_new = []
            subject_proposals_new = [] # uncomment for geometric feats
            object_proposals_new = [] # uncomment for geometric feats
            for i in range(bs):
                x_1_new.append(x_1[i, subj_indices[i], :].unsqueeze(0))
                x_2_new.append(x_2[i, obj_indices[i], :].unsqueeze(0))
                subject_proposals_new.append(subject_proposals[i, subj_indices[i]].unsqueeze(0)) # uncomment for geometric feats
                object_proposals_new.append(object_proposals[i, obj_indices[i]].unsqueeze(0)) # uncomment for geometric feats

            x_1 = torch.cat(x_1_new, dim=0)
            x_2 = torch.cat(x_2_new, dim=0)
            subject_proposals = torch.cat(subject_proposals_new, dim=0) # uncomment for geometric feats
            object_proposals = torch.cat(object_proposals_new, dim=0) # uncomment for geometric feats
        
        if self.training:
            if self.cos_feats:
                geo_feats = extract_multi_position_matrix_nd(subject_proposals, object_proposals)

                geo_feats_list = []
                for i in range(bs):
                    a = extract_pairwise_multi_position_embedding_nd(geo_feats[i])
                    geo_feats_list.append(a.unsqueeze(0))

                geo_feats = torch.cat(geo_feats_list, dim=0)
            else:
                geo_feats = get_proposal_feat(subject_proposals, object_proposals)
            geo_feats = self.geo_project_v2(geo_feats)

            rels = rels.unsqueeze(1).unsqueeze(1).repeat(1,self.num_prop, self.num_prop,1)

            x_1 = self.image2rel_s(x_1)
            x_2 = self.image2rel_o(x_2)
            
            
            x_2 = x_2.unsqueeze(1).repeat(1, self.num_prop,1,1)
            x_1 = x_1.unsqueeze(2).repeat(1,1,self.num_prop,1)

                    
            geo_feats = geo_feats.view(bs, self.num_prop, self.num_prop, -1)

            
            pair_score = torch.cat([x_1, x_2, geo_feats], dim=-1)
            pair_score = self.pair_fuse_feats(pair_score.view(bs, self.num_prop*self.num_prop, -1))
            
            
            pair_score = pair_score.view(bs, self.num_prop, self.num_prop, -1)

            pair_score = (torch.nn.functional.cosine_similarity(pair_score, rels , dim=-1) +1)/2# Uncomment for coisne
            
            pair_score = pair_score.view(bs, self.num_prop, self.num_prop)
        else:
            pair_score=None


        x_subj_test = torch.clamp(x_subj_test, max = 0.99)
        x_subj_test = x_subj_test.view(-1) + 0.0000001

        x_obj_test = torch.clamp(x_obj_test, max = 0.99)
        x_obj_test = x_obj_test.view(-1) + 0.0000001

        nodes_score_ = torch.clamp(nodes_score_, max = 0.99)
        nodes_score_ = nodes_score_.view(-1) + 0.0000001


        x_subj = torch.cat([1-x_subj_test.unsqueeze(-1), x_subj_test.unsqueeze(-1)], dim =-1)
        x_obj = torch.cat([1-x_obj_test.unsqueeze(-1), x_obj_test.unsqueeze(-1)], dim =-1)

        nodes_score_ = torch.cat([1-nodes_score_.unsqueeze(-1), nodes_score_.unsqueeze(-1)], dim =-1)

        x_subj = x_subj.view(-1, 2)
        x_obj = x_obj.view(-1,2)
        x_subj = x_subj.view(bs,orig_num_prop, 2)
        x_obj = x_obj.view(bs, orig_num_prop,2)
        nodes_score_ = nodes_score_.view(-1, orig_num_prop,2)
        
        bbox_deltas = self.bbox_pred(x)
        
        return x_subj,x_obj, pair_score, bbox_deltas, [subj_indices, obj_indices], [pair_score_full], shapes, edges, nodes_score_, node_shapes


model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
}


def fasterrcnn_resnet50_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):
    """
    Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with values of ``x``
          between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values of ``x``
          between ``0`` and ``W`` and values of ``y`` between ``0`` and ``H``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction

    Faster R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.

    Example::

        >>> model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        >>> # For training
        >>> images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
        >>> labels = torch.randint(1, 91, (4, 11))
        >>> images = list(image for image in images)
        >>> targets = []
        >>> for i in range(len(images)):
        >>>     d = {}
        >>>     d['boxes'] = boxes[i]
        >>>     d['labels'] = labels[i]
        >>>     targets.append(d)
        >>> output = model(images, targets)
        >>> # For inference
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    # check default parameters and by default set it to 3 if possible
    trainable_backbone_layers = _validate_resnet_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers)

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)

    # backbone = resnet_backbone('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes, **kwargs)

    print("Loading FPN model")
    state_dict = torch.load("/path/to/model/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth")
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)                                      
    # model.load_state_dict(state_dict)
    overwrite_eps(model, 0.0)
    return model
