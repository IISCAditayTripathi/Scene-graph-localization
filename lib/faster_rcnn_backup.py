from collections import OrderedDict

import torch
from torch import fake_quantize_per_tensor_affine, nn, squeeze, unsqueeze
import torch.nn.functional as F
from torch.nn.modules import linear
from torch.nn.modules.distance import CosineSimilarity

from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign

from ._utils import overwrite_eps
#from ..utils import load_state_dict_from_url
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from .anchor_utils import AnchorGenerator
from .generalized_rcnn import GeneralizedRCNN
from .rpn import RPNHead, RegionProposalNetwork
from .roi_heads import RoIHeads
# from .roi_head_full import RoIHeads

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

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        # if box_head is None:
        #     resolution = box_roi_pool.output_size[0]
        #     representation_size = 1024
        #     box_head = TwoMLPHead(
        #         out_channels * resolution ** 2,
        #         representation_size)

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

        # if box_predictor is None:
        #     representation_size = 1024
        #     # representation_size = 256
        #     box_predictor = FastRCNNPredictor(
        #         representation_size,
        #         num_classes)

        # if box_predictor is None:
        if True:
            # representation_size = 1024
            representation_size = 256
            box_predictor = FastRCNNPredictorJoint(
                representation_size,
                num_classes)


        roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)


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

# class TwoCNNHead(nn.Module):
#     def __init__(self, in_channels, representation_size) -> None:
#         super(TwoCNNHead, self).__init__()
#         self.conv6 = nn.Conv2d(in_channels=in_channels, out_channels=representation_size, kernel_size=1, stride=1, padding=0, bias=True)
#         self.conv7 = nn.Conv2d(in_channels=representation_size, out_channels=representation_size, kernel_size=1, stride=1, padding=0, bias=True)
#         self.bn_1 = nn.BatchNorm2d(representation_size)
#         self.bn_2 = nn.BatchNorm2d(representation_size)
#     def forward(self, x):
#         x = self.conv6(x)
#         x = self.bn_1(x)
#         x = F.relu(x)
#         x = self.conv7(x)
#         x = self.bn_2(x)
#         # x = F.relu(x)
#         x = x.mean(-1).mean(-1)
#         return x

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



# class FastRCNNPredictor(nn.Module):
#     """
#     Standard classification + bounding box regression layers
#     for Fast R-CNN.

#     Args:
#         in_channels (int): number of input channels
#         num_classes (int): number of output classes (including background)
#     """

#     def __init__(self, in_channels, num_classes):
#         super(FastRCNNPredictor, self).__init__()
#         self.cls_score = nn.Linear(in_channels, num_classes)
#         self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

#     def forward(self, x):
#         subject = x[1]
        
#         if x.dim() == 4:
#             assert list(x.shape[2:]) == [1, 1]
#         x = x.flatten(start_dim=1)
#         scores = self.cls_score(x)
#         bbox_deltas = self.bbox_pred(x)

#         return scores, bbox_deltas

class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        in_channels = 1024
        self.cls_score = nn.Linear(2*in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, (num_classes+1) * 4)

        self.linear_nodes = nn.Linear(in_channels, 128)
        self.linear_relation = nn.Linear(in_channels, 128)

        self.linear_nodes_query = nn.Linear(in_channels, 128)
        self.linear_relation_query = nn.Linear(in_channels, 128)

        self.lmda = 0.1
        train = 128
        test = 300
        self.num_prop = test
        # print(self.training)
        # if self.training:
        #     self.num_prop = 128
        # else:
        #     self.num_prop = 1000
        self.cos = CosineSimilarity(dim=-1)
        # self.cls_score = nn.Linear(in_channels, num_classes)
        # self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        subject = x[1]
        object = x[2]
        relation = x[3]
        x = x[0]
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]

        # x = x.flatten(start_dim=1)
        n_dim = 1024 # 256 before
        bs = relation.shape[0]

        
        # ''' # Message Passing in proposals
        x_s = self.linear_nodes(x)
        x_o = self.linear_nodes(x)

        rel = self.linear_relation(relation)
        
        x_s = x_s.view(bs, self.num_prop, 128)
        x_o = x_o.view(bs, self.num_prop, 128)

        x_o = rel.unsqueeze(1)*x_o
        intermediate = torch.bmm(x_s, x_o.permute(0,2,1))
        x_r = intermediate/11.37
        x_r = F.softmax(x_r, dim=-1)
        x_r = torch.bmm(x_r, x.view(bs, self.num_prop, n_dim))

        x = x+ 0.1*x_r.view(-1, n_dim)
        # ''' # Message Passing in proposals
        
        # ''' # Message passing in queries
        ob = self.linear_relation_query(relation)*self.linear_nodes_query(object)
        sc = torch.bmm(self.linear_nodes_query(subject).unsqueeze(1), ob.unsqueeze(-1))
        
        subject = subject + 0.1*sc.squeeze(1)*object
        object = object + 0.1*sc.squeeze(1)*subject
        # ''' # Message passing in queries


        subject = subject.unsqueeze(1).repeat(1,self.num_prop,1)
        subject = subject.view(-1,n_dim)
        
        object = object.unsqueeze(1).repeat(1,self.num_prop,1)
        object = object.view(-1, n_dim)


        # x_subj = self.cos(x,subject) #-----------------------------------------------------------|
        # x_obj = self.cos(x, object)                                                   #          |
        #                                                                             ##               -> uncomment for cosine loss
        # x_subj = torch.cat([1-x_subj.unsqueeze(-1), x_subj.unsqueeze(-1)], dim =-1)#             |
        # x_obj = torch.cat([1-x_obj.unsqueeze(-1), x_obj.unsqueeze(-1)], dim =-1)#----------------|
        x_subj = self.cls_score(torch.cat([x, subject], dim=-1)) # uncomment for CE
        x_obj = self.cls_score(torch.cat([x, object], dim=-1))   # uncomment for CE

        

        
        '''
        subject = subject.view(bs,512, 256)
        object = object.view(bs,512,256)

        subject = subject.unsqueeze(1).repeat(1,512,1,1)
        object = object.unsqueeze(2).repeat(1,1,512,1)
        rel = subject-object
        rel = rel.view(bs,-1,256)

        relation_score = torch.bmm(rel, relation.unsqueeze(-1))
        '''
        bbox_deltas = self.bbox_pred(x)

        return x_subj,x_obj, intermediate , bbox_deltas

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
    # s_rois = s_rois.view(-1, 4)
    # o_roi = o_roi.view(-1, 4)
    # Enumerate all roi combinations, and output the features
    # roisS = s_rois.unsqueeze(1).expand(-1, s_rois.shape[0], -1).reshape(-1, 4)
    # roisO = o_roi.unsqueeze(0).expand(o_roi.shape[0], -1, -1).reshape(-1, 4)
    # roisS = s_rois.unsqueeze(1).expand(-1, num_props, -1).reshape(-1, 4)
    # roisO = o_roi.unsqueeze(0).expand(num_props, -1, -1).reshape(-1, 4)
    # x_1 = x_1.unsqueeze(1).repeat(1, self.num_prop,1,1)
    # x_2 = x_2.unsqueeze(2).repeat(1,1,self.num_prop,1)
    # roisS = s_rois.unsqueeze(1).repeat(1, num_props, 1, 1).reshape(-1, 4)
    # roisO = o_roi.unsqueeze(2).repeat(1,1,num_props, 1).reshape(-1, 4)

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

class FastRCNNPredictorRel(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictorRel, self).__init__()
        in_channels = 1024
        n_dim = 556
        self.bbox_pred = nn.Linear(in_channels, (num_classes+1) * 4)
        self.bbox_pred_rel = nn.Linear(in_channels, (num_classes+1) * 4)

        self.subj_proj = nn.Linear(n_dim, in_channels)
        self.obj_proj = nn.Linear(n_dim, in_channels)
        self.theta_rel = nn.Sequential(
                                    nn.Linear(n_dim, in_channels)
         )

        self.cls_score_s = nn.Linear(2*in_channels, 1) 

        self.cls_score_o = nn.Linear(2*in_channels, 1)

        self.is_cosine = False

        if not self.is_cosine:

            self.cls_score_subj = nn.Linear(2*in_channels, 2) 

            self.cls_score_obj = nn.Linear(2*in_channels, 2)

            self.cls_score_rel = nn.Linear(2*in_channels, 2)



    def forward(self, x):
        subject = x[1]
        object = x[2]
        relation = x[3]
        proposals = x[4]
        # labels = x[5]
        

        proposals_rel = x[5]
        x_relation = x[-1]
        x = x[0]
        prop = []
        for proposal in proposals:
            prop.append(proposal.unsqueeze(0))
        prop = torch.cat(prop, dim=0)

        prop_rel = []
        for proposal in proposals_rel:
            prop_rel.append(proposal.unsqueeze(0))
        prop_rel = torch.cat(prop_rel, dim=0)
        self.num_prop = prop.shape[1]
        self.num_prop_rel = prop_rel.shape[1]
        n_dim = 1024
        bs = relation.shape[0]
        relation = self.theta_rel(relation)

        subject = self.subj_proj(subject)
        object = self.obj_proj(object)

        subject = subject.unsqueeze(1).repeat(1,self.num_prop,1)
        object = object.unsqueeze(1).repeat(1,self.num_prop,1)

        relation = relation.unsqueeze(1).repeat(1,self.num_prop_rel,1)
        
        subject = subject.permute(2,0,1)
        object = object.permute(2,0,1)
        relation = relation.permute(2,0,1)

        subject = subject.view(n_dim, -1)
        object = object.view(n_dim, -1)
        relation = relation.view(n_dim, -1)

        subject = subject.permute(1,0)
        object = object.permute(1,0)
        relation = relation.permute(1,0)
        if self.is_cosine:
            x_subj = (torch.nn.functional.cosine_similarity(x, subject, dim=-1) +1)/2# Uncomment for coisne
            x_obj = (torch.nn.functional.cosine_similarity(x, object, dim=-1) +1)/2# Uncomment for coisne
            x_rel = (torch.nn.functional.cosine_similarity(x_relation, relation, dim=-1)+1)/2

        else:
            x_subj = torch.cat([x,subject], dim=-1)
            # x_subj = self.proj(x_subj)
            x_subj = self.cls_score_subj(x_subj)

            x_obj = torch.cat([x, object], dim=-1)
            # x_obj = self.proj(x_obj)
            x_obj = self.cls_score_obj(x_obj)

            x_rel = torch.cat([x_relation, relation], dim=-1)
            x_rel = self.cls_score_rel(x_rel)
            
            x_subj = F.softmax(x_subj, dim=-1)
            x_obj = F.softmax(x_obj, dim=-1)
            x_rel = F.softmax(x_rel, dim=-1)


        if self.is_cosine:
            x_subj = x_subj.view(bs, self.num_prop)
            x_obj = x_obj.view(bs, self.num_prop)
            x_rel = x_rel.view(bs, self.num_prop_rel)
            x_subj = torch.clamp(x_subj, max = 0.99)
            x_subj = x_subj.view(-1) + 0.0000001

            x_obj = torch.clamp(x_obj, max = 0.99)
            x_obj = x_obj.view(-1) + 0.0000001

            x_rel = torch.clamp(x_rel, max = 0.99)
            x_rel = x_rel.view(-1) + 0.0000001


            x_subj = torch.cat([1-x_subj.unsqueeze(-1), x_subj.unsqueeze(-1)], dim =-1)
            x_obj = torch.cat([1-x_obj.unsqueeze(-1), x_obj.unsqueeze(-1)], dim =-1)
            x_rel = torch.cat([1-x_rel.unsqueeze(-1), x_rel.unsqueeze(-1)], dim =-1)

        
        x_subj = x_subj.view(-1, 2)
        x_obj = x_obj.view(-1,2)
        x_rel = x_rel.view(-1, 2)
        x = x.flatten(start_dim=1)

        bbox_deltas = self.bbox_pred(x)

        bbox_deltas_rel = self.bbox_pred_rel(x_relation)

        return x_subj, x_obj, x_rel, bbox_deltas, bbox_deltas_rel


class FastRCNNPredictorPairedSorted(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictorPairedSorted, self).__init__()
        in_channels = 1024
        n_dim = 556
        self.subj_proj = nn.Linear(n_dim, in_channels)
        self.obj_proj = nn.Linear(n_dim, in_channels)
        self.cls_score = nn.Linear(2*in_channels, 1) 

        self.cls_score_o = nn.Linear(2*in_channels, 1) 
        self.image2rel_s = nn.Linear(in_channels, in_channels)
        self.image2rel_o = nn.Linear(in_channels, in_channels)
        # self.cls_score_v2 = nn.Linear(128, 1)
        # self.proj = nn.Sequential(nn.Linear(2*in_channels, 512),
        #                             nn.ReLU(),
        #                             nn.Linear(512, 128)
        #                             )
        # self.cls_score_pair_v2 = nn.Linear(3*in_channels+64, 1)
        self.is_cosine = True

        if not self.is_cosine:

            self.cls_pair_proj_v3 = nn.Sequential(
                                        nn.Linear(4*in_channels, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 128),
                                        nn.ReLU()
            )
            self.theta_rel_subj = nn.Sequential(
                                    nn.Linear(n_dim, in_channels)
             )

            self.theta_rel_obj = nn.Sequential(
                                        nn.Linear(n_dim, in_channels)
             )

        else:

            self.cls_pair_proj_v2 = nn.Sequential(
                                        nn.Linear(2*in_channels, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 128),
                                        nn.ReLU()
            )

        self.cls_score_pair = nn.Linear(128, 1)

        if True:
            self.pair_fuse_feats = nn.Sequential(
                                        nn.Linear(2*in_channels+64, in_channels),
                                        nn.ReLU(),
                                        nn.Linear(in_channels, in_channels),
                                        nn.ReLU(),
                                        nn.Linear(in_channels, in_channels),

            )

        if False:

            self.pair_fuse_feats = nn.Sequential(
                                        nn.Linear(64, in_channels),
                                        nn.ReLU(),
                                        nn.Linear(in_channels, in_channels),
                                        nn.ReLU(),
                                        nn.Linear(in_channels, in_channels),

            )

            self.pair_fuse_feats_relation = nn.Sequential(
                                        nn.Linear(2*in_channels, in_channels),
                                        nn.ReLU(),
                                        nn.Linear(in_channels, in_channels),
                                        nn.ReLU(),
                                        nn.Linear(in_channels, in_channels),

            )
        # self.cls_score_pair_v2 = nn.Linear(3*in_channels+64, 1)
        # self.pair_proj = nn.Linear(2*in_channels+64, in_channels)

        self.cls_score_pair_geo = nn.Linear(128, 1) # uncomment for geometric feats

        self.bbox_pred = nn.Linear(in_channels, (num_classes+1) * 4)
        # self.theta_rel = nn.Linear(in_channels, in_channels)
        n_dim = 556 # in_channels before
        # self.theta_rel = nn.Sequential(
        #                             nn.Linear(n_dim, in_channels),
        #                             nn.ReLU(),
        #                             nn.Linear(in_channels, in_channels)
        # )

        self.theta_rel = nn.Sequential(
                                    nn.Linear(n_dim, in_channels)
         )
        self.theta_rel_geo = nn.Linear(in_channels, 64) # uncomment for geometric feats
        self.geo_project_v2 = nn.Sequential(nn.Linear(64, 64),
                                            # nn.BatchNorm1d(32),
                                            nn.ReLU(),
                                            nn.Linear(64,64))
        train = 256
        test = 300
        self.sel = train
        self.num_prop = self.sel
        self.finetune = True

    def forward(self, x):
        subject = x[1]
        object = x[2]
        relation = x[3]
        proposals = x[4]
        # labels = x[5]
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
        bs = relation.shape[0]

        # la = la.view(bs, -1)

        relation = self.theta_rel(relation)
        

        if not self.is_cosine:

            subj_rel = self.theta_rel_subj(subject)
            obj_rel = self.theta_rel_obj(object)
        
        # relation_geo = self.theta_rel_geo(relation) # uncomment for geometric feats
        subject = self.subj_proj(subject)
        object = self.obj_proj(object)

        subject = subject.unsqueeze(1).repeat(1,self.num_prop,1)
        object = object.unsqueeze(1).repeat(1,self.num_prop,1)
        
        # x = x.permute(2,0,1)
        subject = subject.permute(2,0,1)
        object = object.permute(2,0,1)

        # x = x.view(n_dim, -1)
        # print(subject.shape)
        subject = subject.view(n_dim, -1)
        object = object.view(n_dim, -1)

        # x = x.permute(1,0)
        subject = subject.permute(1,0)
        object = object.permute(1,0)
        # x = x.view(-1, n_dim)
        # subject = subject.view(-1, n_dim)
        # object = object.view(-1, n_dim)
        
        

        if not self.is_cosine:
            
            x_subj = torch.cat([x,subject], dim=-1)
            # x_subj = self.proj(x_subj)
            x_subj = self.cls_score(x_subj)

            x_obj = torch.cat([x, object], dim=-1)
            # x_obj = self.proj(x_obj)
            x_obj = self.cls_score_o(x_obj)
            
            x_subj = F.sigmoid(x_subj)
            x_obj = F.sigmoid(x_obj)
        if self.is_cosine:
            x_subj = (torch.nn.functional.cosine_similarity(x, subject, dim=-1) +1)/2# Uncomment for coisne
            x_obj = (torch.nn.functional.cosine_similarity(x, object, dim=-1) +1)/2# Uncomment for coisne
            # print(x_subj)

        
        
        x_subj = x_subj.view(bs, self.num_prop)
        x_obj = x_obj.view(bs, self.num_prop)
        
        x_subj_test = x_subj.clone()
        x_obj_test = x_obj.clone()

        x_subj, subj_indices = torch.sort(x_subj, dim=-1, descending=True)
        x_obj, obj_indices = torch.sort(x_obj, dim=-1, descending=True)
        
        
        # subj_indices = [torch.randperm(x_subj.shape[-1])[0:64] for i in range(bs)]
        # obj_indices = [torch.randperm(x_obj.shape[-1])[0:64] for i in range(bs)]

        # subj_indices = torch.cat(subj_indices, dim=0)
        # obj_indices = torch.cat(obj_indices, dim=0)
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


        # x_subj = x_subj[subj_indices]
        # x_obj = x_obj[obj_indices]
        

        
        
        x_1 = x.view(bs, self.num_prop, n_dim)
        x_2 = x.view(bs, self.num_prop, n_dim)

        if self.finetune:
            x_1_full = x.clone().view(bs, self.num_prop, n_dim)
            x_2_full = x.clone().view(bs, self.num_prop, n_dim)            

        subject_proposals = prop.clone() # uncomment for geometric feats
        object_proposals = prop.clone() # uncomment for geometric feats


        if self.finetune:
            subject_proposals_full = prop.clone() # uncomment for geometric feats
            object_proposals_full = prop.clone() # uncomment for geometric feats
        
            proposals_full = self.num_prop

        if self.training:
        # if False:
            if True:
                subject_new = []
                object_new = []
                for i in range(bs):
                        subject_new.append(x_subj[i, subj_indices[i]].unsqueeze(0))
                        object_new.append(x_obj[i, obj_indices[i]].unsqueeze(0))

                x_subj = torch.cat(subject_new, dim=0)
                x_obj = torch.cat(object_new, dim=0)
            nen = 48
            x_subj= x_subj[:,0:nen]
            # x_subj_l = x_subj[:, 90:90+nen//2]
            x_obj = x_obj[:,0:nen]
            # x_obj_l = x_obj[:,90:90+nen//2]

            subj_indices = subj_indices[:,0:nen]
            # subj_indices_l = subj_indices[:,90:90+nen//2]
            obj_indices = obj_indices[:, 0:nen]
            # obj_indices_l = obj_indices[:, 90:90+nen//2]
            self.num_prop = nen

            # x_subj = torch.cat([x_subj_u, x_subj_l], dim=1)
            # x_obj = torch.cat([x_obj_u, x_obj_l], dim=1)
            # subj_indices = torch.cat([subj_indices_u, subj_indices_l], dim=1)
            # obj_indices = torch.cat([obj_indices_u, obj_indices_l], dim=1)
            # print(x_subj)

        bs = x_1.shape[0]
        if self.finetune:
            subject_proposals_full = subject_proposals.clone()
            object_proposals_full = object_proposals.clone()
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
        

        # geo_feats = get_proposal_feat(subject_proposals, object_proposals) # uncomment for geometric feats
        geo_feats = extract_multi_position_matrix_nd(subject_proposals, object_proposals)

        geo_feats_list = []
        for i in range(bs):
            a = extract_pairwise_multi_position_embedding_nd(geo_feats[i])
            geo_feats_list.append(a.unsqueeze(0))

        geo_feats = torch.cat(geo_feats_list, dim=0)
        geo_feats = self.geo_project_v2(geo_feats)

        if self.finetune:
            geo_feats_full = extract_multi_position_matrix_nd(subject_proposals_full, object_proposals_full)

            geo_feats_list = []
            for i in range(bs):
                a = extract_pairwise_multi_position_embedding_nd(geo_feats_full[i])
                geo_feats_list.append(a.unsqueeze(0))

            geo_feats_full = torch.cat(geo_feats_list, dim=0)
            geo_feats_full = self.geo_project_v2(geo_feats_full)

        relation_full = relation.clone()
        relation = relation.unsqueeze(1).unsqueeze(1).repeat(1,self.num_prop, self.num_prop,1)

        if self.finetune:
            relation_full = relation_full.clone().unsqueeze(1).unsqueeze(1).repeat(1,proposals_full, proposals_full,1)

        if not self.is_cosine:

            subj_rel = subj_rel.unsqueeze(1).unsqueeze(1).repeat(1,self.num_prop, self.num_prop,1)

            obj_rel = obj_rel.unsqueeze(1).unsqueeze(1).repeat(1,self.num_prop, self.num_prop,1)

        # relation_geo = relation_geo.unsqueeze(1).unsqueeze(1).repeat(1,self.num_prop, self.num_prop,1)# uncomment for geometric feats
        x_1 = self.image2rel_s(x_1)
        x_2 = self.image2rel_o(x_2)
        
        
        # x_1 = x_1.unsqueeze(1).repeat(1, self.num_prop,1,1)
        # x_2 = x_2.unsqueeze(2).repeat(1,1,self.num_prop,1)
        x_2 = x_2.unsqueeze(1).repeat(1, self.num_prop,1,1)
        x_1 = x_1.unsqueeze(2).repeat(1,1,self.num_prop,1)

        if self.finetune:
            x_1_full = self.image2rel_s(x_1_full)
            x_2_full = self.image2rel_o(x_2_full)
            
            
            # x_1 = x_1.unsqueeze(1).repeat(1, self.num_prop,1,1)
            # x_2 = x_2.unsqueeze(2).repeat(1,1,self.num_prop,1)
            x_2_full = x_2_full.unsqueeze(1).repeat(1, proposals_full,1,1)
            x_1_full = x_1_full.unsqueeze(2).repeat(1,1,proposals_full,1)
        
        
        geo_feats = geo_feats.view(bs, self.num_prop, self.num_prop, -1)
        if self.finetune:
            geo_feats_full = geo_feats_full.view(bs, proposals_full, proposals_full, -1)

        # pair_score_geo = torch.cat([geo_feats, relation_geo], dim=-1)# uncomment for geometric feats
        # pair_score_geo = self.cls_score_pair_geo(pair_score_geo.view(bs, self.num_prop*self.num_prop, -1))# uncomment for geometric feats
        # pair_score_geo = pair_score_geo.view(bs, self.num_prop, self.num_prop)# uncomment for geometric feats
        # pair_score_geo = F.sigmoid(pair_score_geo)# uncomment for geometric feats

        # pair = torch.cat([x_1, x_2, geo_feats], dim=-1)
        # pair = pair.view(bs, self.num_prop*self.num_prop, -1)
        # pair = self.pair_proj(pair)
        # pair = pair.view(bs, self.num_prop, self.num_prop, -1)
        # pair_score = torch.cat([x_1, x_2, relation, geo_feats], dim=-1)
        
        if True:
            pair_score = torch.cat([x_1, x_2, geo_feats], dim=-1)
            # pair_score = torch.cat([x_1, x_2], dim=-1)
            pair_score = self.pair_fuse_feats(pair_score.view(bs, self.num_prop*self.num_prop, -1))

            if self.finetune:
                # pair_score_full = geo_feats_full
                pair_score_full = torch.cat([x_1_full, x_2_full, geo_feats_full], dim=-1)
                # pair_score_full = torch.cat([x_1_full, x_2_full], dim=-1)
                pair_score_full = self.pair_fuse_feats(pair_score_full.view(bs, proposals_full*proposals_full, -1))

        if False:
            pair_score = geo_feats
            pair_score = self.pair_fuse_feats(pair_score.view(bs, self.num_prop*self.num_prop, -1))
            pair_score_box = torch.cat([x_1, x_2], dim=-1)
            pair_score_box = self.pair_fuse_feats_relation(pair_score_box.view(bs, self.num_prop*self.num_prop, -1))

            if self.finetune:
                pair_score_full = geo_feats_full
                pair_score_full = self.pair_fuse_feats(pair_score_full.view(bs, proposals_full*proposals_full, -1))

                pair_score_box_full = torch.cat([x_1_full, x_2_full], dim=-1)
                pair_score_box_full = self.pair_fuse_feats_relation(pair_score_box_full.view(bs, proposals_full*proposals_full, -1))
        
        
        pair_score = pair_score.view(bs, self.num_prop, self.num_prop, -1)

        if self.finetune:
            pair_score_full = pair_score_full.view(bs, proposals_full, proposals_full, -1)
         
        # print(pair_score.shape)   
        if self.is_cosine:   
            if False: 
                pair_score = (torch.nn.functional.cosine_similarity(pair_score, relation, dim=-1) +1)/2# Uncomment for coisne
            else:
                # pair_score_box = pair_score_box.view(bs, self.num_prop, self.num_prop, -1)
                pair_score = (torch.nn.functional.cosine_similarity(pair_score, relation, dim=-1) +1)/2# Uncomment for coisne
                # pair_score_box = (torch.nn.functional.cosine_similarity(pair_score_box, relation, dim=-1) +1)/2# Uncomment for coisne

                if self.finetune:
                    # pair_score_box_full = pair_score_box_full.view(bs, proposals_full, proposals_full, -1)
                    pair_score_full = (torch.nn.functional.cosine_similarity(pair_score_full, relation_full, dim=-1) +1)/2# Uncomment for coisne
                    # pair_score_box_full = (torch.nn.functional.cosine_similarity(pair_score_box_full, relation_full, dim=-1) +1)/2# Uncomment for coisne

        if not self.is_cosine:
            pair_score = torch.cat([pair_score, relation, subj_rel, obj_rel], dim=-1)
            # pair_score = torch.cat([pair, relation], dim=-1)
            pair_score = self.cls_pair_proj_v3(pair_score.view(bs, self.num_prop*self.num_prop, -1))

            pair_score = self.cls_score_pair(pair_score)
        
        pair_score = pair_score.view(bs, self.num_prop, self.num_prop)
        if self.finetune:
            pair_score_full = pair_score_full.view(bs, proposals_full, proposals_full)

        # if True:
            # pair_score_box = pair_score_box.view(bs, self.num_prop, self.num_prop)
            # if self.finetune:
                # pair_score_box_full = pair_score_box_full.view(bs, proposals_full, proposals_full)

        if not self.is_cosine:
            pair_score = F.sigmoid(pair_score)
        
        if False:
            pair_score = pair_score*x_obj_test.unsqueeze(1)

            pair_score = pair_score*x_subj_test.unsqueeze(-1)
            # print(torch.max(pair_score))
            # pair_score = pair_score*x_obj.unsqueeze(1)

            # pair_score = pair_score*x_subj.unsqueeze(-1)

        x_subj_test = torch.clamp(x_subj_test, max = 0.99)
        x_subj_test = x_subj_test.view(-1) + 0.0000001

        x_obj_test = torch.clamp(x_obj_test, max = 0.99)
        x_obj_test = x_obj_test.view(-1) + 0.0000001


        x_subj = torch.cat([1-x_subj_test.unsqueeze(-1), x_subj_test.unsqueeze(-1)], dim =-1)
        x_obj = torch.cat([1-x_obj_test.unsqueeze(-1), x_obj_test.unsqueeze(-1)], dim =-1)

        x_subj = x_subj.view(-1, 2)
        x_obj = x_obj.view(-1,2)

        bbox_deltas = self.bbox_pred(x)
        # if self.finetune:
        if False:
            return x_subj,x_obj,pair_score_box, pair_score, bbox_deltas, [subj_indices, obj_indices], [pair_score_box_full, pair_score_full]
        elif True:
            return x_subj,x_obj, pair_score, bbox_deltas, [subj_indices, obj_indices], [pair_score_full]
        else:
            if False:
                return x_subj,x_obj,pair_score_box, pair_score, bbox_deltas, [subj_indices, obj_indices]
            if True:
                return x_subj,x_obj,pair_score, bbox_deltas, [subj_indices, obj_indices], None


class match_block(nn.Module):
    def __init__(self, inplanes):
        super(match_block, self).__init__()

        self.sub_sample = False

        self.in_channels = inplanes
        self.inter_channels = None

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.inter_channels = self.in_channels

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d


        self.g = torch.nn.Linear(self.in_channels, self.inter_channels)

        
        self.W = nn.Linear(self.inter_channels, self.in_channels)

        # nn.init.kaiming_normal_(self.W.weight, 0)
        # nn.init.kaiming_normal_(self.W.bias, 0)


        self.theta = torch.nn.Linear(self.in_channels, self.inter_channels)

        self.phi = torch.nn.Linear(self.in_channels, self.inter_channels)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )
        

        
    def forward(self, detect, aim):

        

        batch_size, num_prop, feat_dim = aim.shape


        #####################################find aim image similar object ####################################################

        d_x = self.g(detect).view(batch_size, self.inter_channels, -1)
        d_x = d_x.permute(0, 2, 1).contiguous()

        a_x = self.g(aim).view(batch_size, self.inter_channels, -1)
        a_x = a_x.permute(0, 2, 1).contiguous()

        theta_x = self.theta(aim).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(detect).view(batch_size, self.inter_channels, -1)

        

        f = torch.matmul(theta_x, phi_x)

        N = f.size(-1)
        f_div_C = f / N

        f = f.permute(0, 2, 1).contiguous()
        N = f.size(-1)
        fi_div_C = f / N

        non_aim = torch.matmul(f_div_C, d_x)
        non_aim = non_aim.permute(0, 2, 1).contiguous()
        non_aim = non_aim.view(batch_size, num_prop, feat_dim)
        non_aim = self.W(non_aim)
        non_aim = non_aim + aim


        return non_aim, aim


class cross_block(nn.Module):
    def __init__(self, inplanes):
        super(cross_block, self).__init__()

        self.sub_sample = False

        self.in_channels = inplanes
        self.inter_channels = None

        if self.inter_channels is None:
            self.inter_channels = self.in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        # self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                  kernel_size=1, stride=1, padding=0)

        self.g = nn.Linear(self.in_channels, self.in_channels)
        self.g2 = nn.Linear(self.in_channels, self.in_channels)

        self.W = nn.Linear(self.in_channels, self.in_channels)

        self.Q = nn.Linear(self.in_channels, self.in_channels)

        self.theta = nn.Linear(self.in_channels, self.in_channels)
        self.phi = nn.Linear(self.in_channels, self.in_channels)


        
    def forward(self, detect, aim):

        

        batch_size, num_prop_aim, feat_dim = aim.shape
        batch_size, num_prop_detect, feat_dim = detect.shape


        #####################################find aim image similar object ####################################################

        d_x = self.g(detect).view(batch_size, self.inter_channels, -1)
        d_x = d_x.permute(0, 2, 1).contiguous()

        a_x = self.g2(aim).view(batch_size, self.inter_channels, -1)
        a_x = a_x.permute(0, 2, 1).contiguous()

        theta_x = self.theta(aim).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(detect).view(batch_size, self.inter_channels, -1)

        

        f = torch.matmul(theta_x, phi_x)

        N = f.size(-1)
        f_div_C = f / N

        f = f.permute(0, 2, 1).contiguous()
        N = f.size(-1)
        fi_div_C = f / N

        non_aim = torch.matmul(f_div_C, d_x)
        non_aim = non_aim.permute(0, 2, 1).contiguous()
        non_aim = non_aim.view(batch_size, num_prop_aim, feat_dim)
        non_aim = self.W(non_aim)
        non_aim = non_aim + aim

        non_det = torch.matmul(fi_div_C, a_x)
        non_det = non_det.permute(0, 2, 1).contiguous()
        non_det = non_det.view(batch_size, num_prop_detect, feat_dim)
        non_det = self.Q(non_det)
        non_det = non_det + detect

        return non_det, non_aim

class FastRCNNPredictorPairedSortedAtten(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictorPairedSortedAtten, self).__init__()
        in_channels = 1024
        n_dim = 556
        self.subj_proj = nn.Linear(n_dim, in_channels)
        self.obj_proj = nn.Linear(n_dim, in_channels)
        self.cls_score = nn.Linear(2*in_channels, 1) 

        self.cls_score_o = nn.Linear(2*in_channels, 1) 
        self.image2rel_s = nn.Linear(in_channels, in_channels)
        self.image2rel_o = nn.Linear(in_channels, in_channels)
        # self.cls_score_v2 = nn.Linear(128, 1)
        # self.proj = nn.Sequential(nn.Linear(2*in_channels, 512),
        #                             nn.ReLU(),
        #                             nn.Linear(512, 128)
        #                             )
        # self.cls_score_pair_v2 = nn.Linear(3*in_channels+64, 1)
        self.is_cosine = True

        self.self_attention = match_block(in_channels)

        self.co_attention = cross_block(in_channels)

        if not self.is_cosine:

            self.cls_pair_proj_v3 = nn.Sequential(
                                        nn.Linear(4*in_channels, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 128),
                                        nn.ReLU()
            )
            self.theta_rel_subj = nn.Sequential(
                                    nn.Linear(n_dim, in_channels)
             )

            self.theta_rel_obj = nn.Sequential(
                                        nn.Linear(n_dim, in_channels)
             )

        else:

            self.cls_pair_proj_v2 = nn.Sequential(
                                        nn.Linear(2*in_channels, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 128),
                                        nn.ReLU()
            )

        self.cls_score_pair = nn.Linear(128, 1)

        # self.cls_score_pair_v2 = nn.Linear(3*in_channels+64, 1)
        # self.pair_proj = nn.Linear(2*in_channels+64, in_channels)

        self.cls_score_pair_geo = nn.Linear(128, 1) # uncomment for geometric feats

        self.bbox_pred = nn.Linear(in_channels, (num_classes+1) * 4)
        
        n_dim = 556 # in_channels before
        

        self.theta_rel = nn.Sequential(
                                    nn.Linear(n_dim, in_channels)
         )
        self.theta_rel_geo = nn.Linear(in_channels, 64) # uncomment for geometric feats
        self.geo_project_v2 = nn.Sequential(nn.Linear(64, 64),
                                            # nn.BatchNorm1d(32),
                                            nn.ReLU(),
                                            nn.Linear(64,64))
        train = 256
        test = 300
        self.sel = train
        self.num_prop = self.sel
        self.finetune = True

    def forward(self, x):
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
        bs = relation.shape[0]
        x = x.view(bs, self.num_prop, -1)
        x_box = x.clone()
        subject = self.subj_proj(subject)
        object = self.obj_proj(object)
        relation = self.theta_rel(relation)

        x, _ = self.self_attention(x, x)

        queries = torch.cat([subject.unsqueeze(1), object.unsqueeze(1), relation.unsqueeze(1)], dim=1)

        queries, _ = self.self_attention(queries, queries)

        x, queries = self.co_attention(x, queries)
        
        
        # x, _ = self.self_attention(x, x)

        # queries = torch.cat([subject.unsqueeze(1), object.unsqueeze(1), relation.unsqueeze(1)], dim=1)

        # queries, _ = self.self_attention(queries, queries)

        # x, queries = self.co_attention(x, queries)
        subject, object, relation = queries[:,0,:], queries[:,1,:], queries[:,2,:]


        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        n_dim = 1024 # 256 before
        bs = relation.shape[0]


        # relation = self.theta_rel(relation)
        

        if not self.is_cosine:

            subj_rel = self.theta_rel_subj(subject)
            obj_rel = self.theta_rel_obj(object)
        

        subject = subject.unsqueeze(1).repeat(1,self.num_prop,1)
        object = object.unsqueeze(1).repeat(1,self.num_prop,1)
        
        subject = subject.permute(2,0,1)
        object = object.permute(2,0,1)
        
        # print(subject.shape)
        subject = subject.view(n_dim, -1)
        object = object.view(n_dim, -1)

        # x = x.permute(1,0)
        subject = subject.permute(1,0)
        object = object.permute(1,0)
        x = x.view(-1, n_dim)

        x_box = x_box.view(-1, n_dim)
        
        

        if not self.is_cosine:
            
            x_subj = torch.cat([x,subject], dim=-1)
            # x_subj = self.proj(x_subj)
            x_subj = self.cls_score(x_subj)

            x_obj = torch.cat([x, object], dim=-1)
            # x_obj = self.proj(x_obj)
            x_obj = self.cls_score_o(x_obj)
            
            x_subj = F.sigmoid(x_subj)
            x_obj = F.sigmoid(x_obj)
        if self.is_cosine:
            x_subj = (torch.nn.functional.cosine_similarity(x, subject, dim=-1) +1)/2# Uncomment for coisne
            x_obj = (torch.nn.functional.cosine_similarity(x, object, dim=-1) +1)/2# Uncomment for coisne
            # print(x_subj)

        
        
        x_subj = x_subj.view(bs, self.num_prop)
        x_obj = x_obj.view(bs, self.num_prop)
        
        
        
        x_subj = torch.clamp(x_subj, max = 0.99)
        x_subj = x_subj.view(-1) + 0.0000001

        x_obj = torch.clamp(x_obj, max = 0.99)
        x_obj = x_obj.view(-1) + 0.0000001
        

        x_subj = torch.cat([1-x_subj.unsqueeze(-1), x_subj.unsqueeze(-1)], dim =-1)
        x_obj = torch.cat([1-x_obj.unsqueeze(-1), x_obj.unsqueeze(-1)], dim =-1)

        x_subj = x_subj.view(-1, 2)
        x_obj = x_obj.view(-1,2)

        bbox_deltas = self.bbox_pred(x_box)
        
        return x_subj,x_obj, bbox_deltas
        

class FastRCNNPredictorPairedSorted_v2(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictorPairedSorted_v2, self).__init__()
        in_channels = 1024
        self.cls_score = nn.Linear(2*in_channels, 1)

        self.cls_score_o = nn.Linear(2*in_channels, 1)
        self.image2rel_s = nn.Linear(in_channels, in_channels)
        self.image2rel_o = nn.Linear(in_channels, in_channels)
        # self.cls_score_v2 = nn.Linear(128, 1)
        # self.proj = nn.Sequential(nn.Linear(2*in_channels, 512),
        #                             nn.ReLU(),
        #                             nn.Linear(512, 128)
        #                             )
        # self.cls_score_pair_v2 = nn.Linear(3*in_channels+64, 1)

        self.cls_pair_proj_v2 = nn.Sequential(
                                    nn.Linear(3*in_channels, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 128)
        )

        self.cls_score_pair = nn.Linear(128, 1)
        self.pass_messgae = nn.Sequential(nn.Linear(in_channels*2 +64, in_channels),
                                        nn.ReLU(),
                                        nn.Linear(in_channels, in_channels),
                                        nn.ReLU(),
                                        nn.Linear(in_channels, in_channels)

        )

        self.aggregate = nn.Sequential(nn.Linear(in_channels*2, in_channels),
                                        # nn.BatchNorm1d(in_channels),
                                        nn.ReLU(),
                                        nn.Linear(in_channels, in_channels),
                                        # nn.BatchNorm1d(in_channels),
                                        nn.ReLU(),
                                        nn.Linear(in_channels, in_channels)

        )
        # self.cls_score_pair_v2 = nn.Linear(3*in_channels+64, 1)
        # self.pair_proj = nn.Linear(2*in_channels+64, in_channels)

        self.cls_score_pair_geo = nn.Linear(128, 1) # uncomment for geometric feats

        self.bbox_pred = nn.Linear(in_channels, (num_classes+1) * 4)
        self.theta_rel = nn.Sequential(
                                    nn.Linear(in_channels, in_channels),
                                    nn.ReLU(),
                                    nn.Linear(in_channels, in_channels)
        )
        # self.theta_rel = nn.Linear(in_channels, in_channels)
        # self.theta_rel_geo = nn.Linear(in_channels, 64) # uncomment for geometric feats
        self.geo_project_v2 = nn.Sequential(nn.Linear(64, 64),
                                            # nn.BatchNorm1d(32),
                                            nn.ReLU(),
                                            nn.Linear(64,64),
                                            nn.ReLU(),
                                            nn.Linear(64,64))
        train = 256
        test = 300
        self.sel = train
        self.num_prop = self.sel

    def forward(self, x):
        subject = x[1]
        object = x[2]
        relation = x[3]
        proposals = x[4]
        # labels = x[5]
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
        bs = relation.shape[0]

        # la = la.view(bs, -1)

        relation = self.theta_rel(relation)
        
        # relation_geo = self.theta_rel_geo(relation) # uncomment for geometric feats
        

        subject = subject.unsqueeze(1).repeat(1,self.num_prop,1)
        object = object.unsqueeze(1).repeat(1,self.num_prop,1)
        
        # x = x.permute(2,0,1)
        subject = subject.permute(2,0,1)
        object = object.permute(2,0,1)

        # x = x.view(n_dim, -1)
        subject = subject.view(n_dim, -1)
        object = object.view(n_dim, -1)

        # x = x.permute(1,0)
        subject = subject.permute(1,0)
        object = object.permute(1,0)
        # x = x.view(-1, n_dim)
        # subject = subject.view(-1, n_dim)
        # object = object.view(-1, n_dim)
        # print(subject[0][0:120])
        # print(object[0][0:120])
        # print(x[0][0:120])
        x_box = x.clone()
        x = x.view(bs, self.num_prop, -1)
        x_1 = x.unsqueeze(1).repeat(1, self.num_prop,1,1)
        x_2 = x.unsqueeze(2).repeat(1,1,self.num_prop,1)

        
        subject_proposals = prop.clone() # uncomment for geometric feats
        object_proposals = prop.clone() # uncomment for geometric feats
        # geo_feats = get_proposal_feat(subject_proposals, object_proposals)
        # geo_feats = self.geo_project(geo_feats)
        
        
        geo_feats = extract_multi_position_matrix_nd(subject_proposals, object_proposals)

        geo_feats_list = []
        for i in range(bs):
            a = extract_pairwise_multi_position_embedding_nd(geo_feats[i])
            geo_feats_list.append(a.unsqueeze(0))
        

        geo_feats = torch.cat(geo_feats_list, dim=0)
        geo_feats = self.geo_project_v2(geo_feats)
        geo_feats = geo_feats.view(bs, self.num_prop, self.num_prop, -1)
        

        
        
        fet = torch.cat([x_1, x_2, geo_feats], dim=-1)

        fet = fet.view(bs, self.num_prop*self.num_prop, -1)
        fet = self.pass_messgae(fet)
        fet = fet.view(bs, self.num_prop, self.num_prop, -1)
        # print(fet[0])
        fet, _ = torch.topk(fet, 18, largest=True, dim=2)
        # fet = fet.mean(dim=2)
        fet = fet.sum(dim=2)/8
        x = torch.cat([x, fet], dim=-1)

        x = self.aggregate(x)
        x = x.view(bs*self.num_prop, -1)
        
        
        x_subj = torch.cat([x,subject], dim=-1)
        # x_subj = self.proj(x_subj)
        x_subj = self.cls_score(x_subj)

        x_obj = torch.cat([x, object], dim=-1)
        # x_obj = self.proj(x_obj)
        x_obj = self.cls_score_o(x_obj)
        
        x_subj = F.sigmoid(x_subj)
        x_obj = F.sigmoid(x_obj)
        
        
        x_subj = x_subj.view(bs, self.num_prop)
        x_obj = x_obj.view(bs, self.num_prop)
        
        x_subj_test = x_subj.clone()
        x_obj_test = x_obj.clone()

        x_subj, subj_indices = torch.sort(x_subj, dim=-1, descending=True)
        x_obj, obj_indices = torch.sort(x_obj, dim=-1, descending=True)
        
        
        # subj_indices = [torch.randperm(x_subj.shape[-1])[0:64] for i in range(bs)]
        # obj_indices = [torch.randperm(x_obj.shape[-1])[0:64] for i in range(bs)]

        # subj_indices = torch.cat(subj_indices, dim=0)
        # obj_indices = torch.cat(obj_indices, dim=0)
        if True:
            subj_indices_u = [torch.randperm(40)[0:16] for i in range(bs)]
            obj_indices_u = [torch.randperm(40)[0:16] for i in range(bs)]
            subj_indices_u = torch.cat(subj_indices_u, dim=0)
            obj_indices_u= torch.cat(obj_indices_u, dim=0)

            subj_indices_l = [torch.randperm(67)[0:16]+60 for i in range(bs)]
            obj_indices_l = [torch.randperm(67)[0:16]+60 for i in range(bs)]
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


        # x_subj = x_subj[subj_indices]
        # x_obj = x_obj[obj_indices]
        

        
        
        x_1 = x.view(bs, self.num_prop, n_dim)
        x_2 = x.view(bs, self.num_prop, n_dim)


        if self.training:
            if True:
                subject_new = []
                object_new = []
                for i in range(bs):
                        subject_new.append(x_subj[i, subj_indices[i]].unsqueeze(0))
                        object_new.append(x_obj[i, obj_indices[i]].unsqueeze(0))

                x_subj = torch.cat(subject_new, dim=0)
                x_obj = torch.cat(object_new, dim=0)
            nen = 32
            x_subj= x_subj[:,0:nen]
            # x_subj_l = x_subj[:, 90:90+nen//2]
            x_obj = x_obj[:,0:nen]
            # x_obj_l = x_obj[:,90:90+nen//2]

            subj_indices = subj_indices[:,0:nen]
            # subj_indices_l = subj_indices[:,90:90+nen//2]
            obj_indices = obj_indices[:, 0:nen]
            # obj_indices_l = obj_indices[:, 90:90+nen//2]
            self.num_prop = nen

            # x_subj = torch.cat([x_subj_u, x_subj_l], dim=1)
            # x_obj = torch.cat([x_obj_u, x_obj_l], dim=1)
            # subj_indices = torch.cat([subj_indices_u, subj_indices_l], dim=1)
            # obj_indices = torch.cat([obj_indices_u, obj_indices_l], dim=1)
            # print(x_subj)

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
        

        # geo_feats = get_proposal_feat(subject_proposals, object_proposals) # uncomment for geometric feats
        # geo_feats = extract_multi_position_matrix_nd(subject_proposals, object_proposals)

        # geo_feats_list = []
        # for i in range(bs):
        #     a = extract_pairwise_multi_position_embedding_nd(geo_feats[i])
        #     geo_feats_list.append(a.unsqueeze(0))

        # geo_feats = torch.cat(geo_feats_list, dim=0)


        # geo_feats = self.geo_project(geo_feats)

        relation = relation.unsqueeze(1).unsqueeze(1).repeat(1,self.num_prop, self.num_prop,1)

        # relation_geo = relation_geo.unsqueeze(1).unsqueeze(1).repeat(1,self.num_prop, self.num_prop,1)# uncomment for geometric feats
        x_1 = self.image2rel_s(x_1)
        x_2 = self.image2rel_o(x_2)

        x_1 = x_1.unsqueeze(1).repeat(1, self.num_prop,1,1)
        x_2 = x_2.unsqueeze(2).repeat(1,1,self.num_prop,1)
        
        
        
        geo_feats = geo_feats.view(bs, self.num_prop, self.num_prop, -1)

        # pair_score_geo = torch.cat([geo_feats, relation_geo], dim=-1)# uncomment for geometric feats
        # pair_score_geo = self.cls_score_pair_geo(pair_score_geo.view(bs, self.num_prop*self.num_prop, -1))# uncomment for geometric feats
        # pair_score_geo = pair_score_geo.view(bs, self.num_prop, self.num_prop)# uncomment for geometric feats
        # pair_score_geo = F.sigmoid(pair_score_geo)# uncomment for geometric feats

        pair = torch.cat([x_1, x_2, geo_feats], dim=-1)
        pair = pair.view(bs, self.num_prop*self.num_prop, -1)
        pair = self.pair_proj(pair)
        pair = pair.view(bs, self.num_prop, self.num_prop, -1)
        # pair_score = torch.cat([x_1, x_2, relation], dim=-1)
        pair_score = torch.cat([pair, relation], dim=-1)
        pair_score = self.cls_pair_proj_v2(pair_score.view(bs, self.num_prop*self.num_prop, -1))

        pair_score = self.cls_score_pair(pair_score)
        # pair_score = self.cls_score_pair_v2(pair_score.view(bs, self.num_prop*self.num_prop, -1))
        pair_score = pair_score.view(bs, self.num_prop, self.num_prop)
        pair_score = F.sigmoid(pair_score)
        
        if False:
            pair_score = pair_score*x_obj_test.unsqueeze(1)

            pair_score = pair_score*x_subj_test.unsqueeze(-1)
            # print(torch.max(pair_score))
            # pair_score = pair_score*x_obj.unsqueeze(1)

            # pair_score = pair_score*x_subj.unsqueeze(-1)

        x_subj_test = torch.clamp(x_subj_test, max = 0.99)
        x_subj_test = x_subj_test.view(-1) + 0.000001

        x_obj_test = torch.clamp(x_obj_test, max = 0.99)
        x_obj_test = x_obj_test.view(-1) + 0.000001


        x_subj = torch.cat([1-x_subj_test.unsqueeze(-1), x_subj_test.unsqueeze(-1)], dim =-1)
        x_obj = torch.cat([1-x_obj_test.unsqueeze(-1), x_obj_test.unsqueeze(-1)], dim =-1)

        x_subj = x_subj.view(-1, 2)
        x_obj = x_obj.view(-1,2)


        bbox_deltas = self.bbox_pred(x_box)
        return x_subj,x_obj,pair_score, bbox_deltas, [subj_indices, obj_indices]

class FastRCNNPredictorPairedMP(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictorPairedMP, self).__init__()
        in_channels = 1024
        self.cls_score = nn.Linear(2*in_channels, 1)

        self.cls_score_o = nn.Linear(2*in_channels, 1)
        self.image2rel_s = nn.Linear(in_channels, in_channels)
        self.image2rel_o = nn.Linear(in_channels, in_channels)

        self.self_message = nn.Linear(in_channels, in_channels)
        self.subj2prop = nn.Linear(in_channels, in_channels)
        self.obj2prop = nn.Linear(in_channels, in_channels)
        self.rel2prop = nn.Linear(in_channels, in_channels)
        self.prop2self = nn.Linear(in_channels, in_channels)
        # self.cls_score_v2 = nn.Linear(128, 1)
        # self.proj = nn.Sequential(nn.Linear(2*in_channels, 512),
        #                             nn.ReLU(),
        #                             nn.Linear(512, 128)
        #                             )
        # self.cls_score_pair_v2 = nn.Linear(3*in_channels+64, 1)

        self.cls_pair_proj = nn.Sequential(
                                    nn.Linear(3*in_channels+64, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 128)
        )

        self.cls_score_pair = nn.Linear(128, 1)
        # self.cls_score_pair_v2 = nn.Linear(3*in_channels+64, 1)
        # self.pair_proj = nn.Linear(2*in_channels+64, in_channels)

        self.cls_score_pair_geo = nn.Linear(128, 1) # uncomment for geometric feats

        self.bbox_pred = nn.Linear(in_channels, (num_classes+1) * 4)
        self.theta_rel = nn.Linear(in_channels, in_channels)
        self.theta_rel_geo = nn.Linear(in_channels, 64) # uncomment for geometric feats
        self.geo_project = nn.Sequential(nn.Linear(18, 64),
                                            # nn.BatchNorm1d(32),
                                            nn.ReLU(),
                                            nn.Linear(64,64))
        train = 256
        test = 300
        self.sel = train
        self.num_prop = self.sel

    def forward(self, x):
        subject = x[1]
        object = x[2]
        relation = x[3]
        proposals = x[4]
        # labels = x[5]
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
        bs = relation.shape[0]
        x = x.shape(bs, self.num_prop, -1)

        # la = la.view(bs, -1)

        # relation = self.theta_rel(relation)
        
        relation_geo = self.theta_rel_geo(relation) # uncomment for geometric feats
        

        subject = subject.unsqueeze(1).repeat(1,self.num_prop,1)
        object = object.unsqueeze(1).repeat(1,self.num_prop,1)
        
        # x = x.permute(2,0,1)
        subject = subject.permute(2,0,1)
        object = object.permute(2,0,1)

        # x = x.view(n_dim, -1)
        subject = subject.view(n_dim, -1)
        object = object.view(n_dim, -1)

        # x = x.permute(1,0)
        subject = subject.permute(1,0)
        object = object.permute(1,0)
        
        
        x_subj = torch.cat([x,subject], dim=-1)
        # x_subj = self.proj(x_subj)
        x_subj = self.cls_score(x_subj)

        x_obj = torch.cat([x, object], dim=-1)
        # x_obj = self.proj(x_obj)
        x_obj = self.cls_score_o(x_obj)
        
        x_subj = F.sigmoid(x_subj)
        x_obj = F.sigmoid(x_obj)
        
        
        x_subj = x_subj.view(bs, self.num_prop)
        x_obj = x_obj.view(bs, self.num_prop)
        
        x_subj_test = x_subj.clone()
        x_obj_test = x_obj.clone()

        x_subj, subj_indices = torch.sort(x_subj, dim=-1, descending=True)
        x_obj, obj_indices = torch.sort(x_obj, dim=-1, descending=True)
        if True:
            subj_indices_u = [torch.randperm(40)[0:16] for i in range(bs)]
            obj_indices_u = [torch.randperm(60)[0:16] for i in range(bs)]
            subj_indices_u = torch.cat(subj_indices_u, dim=0)
            obj_indices_u= torch.cat(obj_indices_u, dim=0)

            subj_indices_l = [torch.randperm(140)[0:16]+60 for i in range(bs)]
            obj_indices_l = [torch.randperm(140)[0:16]+80 for i in range(bs)]
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


        # x_subj = x_subj[subj_indices]
        # x_obj = x_obj[obj_indices]
        

        
        
        x_1 = x.view(bs, self.num_prop, n_dim)
        x_2 = x.view(bs, self.num_prop, n_dim)

        subject_proposals = prop.clone() # uncomment for geometric feats
        object_proposals = prop.clone() # uncomment for geometric feats

        if self.training:
            if True:
                subject_new = []
                object_new = []
                for i in range(bs):
                        subject_new.append(x_subj[i, subj_indices[i]].unsqueeze(0))
                        object_new.append(x_obj[i, obj_indices[i]].unsqueeze(0))

                x_subj = torch.cat(subject_new, dim=0)
                x_obj = torch.cat(object_new, dim=0)
            nen = 32
            x_subj= x_subj[:,0:nen]
            # x_subj_l = x_subj[:, 90:90+nen//2]
            x_obj = x_obj[:,0:nen]
            # x_obj_l = x_obj[:,90:90+nen//2]

            subj_indices = subj_indices[:,0:nen]
            # subj_indices_l = subj_indices[:,90:90+nen//2]
            obj_indices = obj_indices[:, 0:nen]
            # obj_indices_l = obj_indices[:, 90:90+nen//2]
            self.num_prop = nen

            # x_subj = torch.cat([x_subj_u, x_subj_l], dim=1)
            # x_obj = torch.cat([x_obj_u, x_obj_l], dim=1)
            # subj_indices = torch.cat([subj_indices_u, subj_indices_l], dim=1)
            # obj_indices = torch.cat([obj_indices_u, obj_indices_l], dim=1)
            # print(x_subj)

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
        

        geo_feats = get_proposal_feat(subject_proposals, object_proposals) # uncomment for geometric feats
        # geo_feats = extract_multi_position_matrix_nd(subject_proposals, object_proposals)

        # geo_feats_list = []
        # for i in range(bs):
        #     a = extract_pairwise_multi_position_embedding_nd(geo_feats[i])
        #     geo_feats_list.append(a.unsqueeze(0))

        # geo_feats = torch.cat(geo_feats_list, dim=0)


        geo_feats = self.geo_project(geo_feats)

        relation = relation.unsqueeze(1).unsqueeze(1).repeat(1,self.num_prop, self.num_prop,1)

        relation_geo = relation_geo.unsqueeze(1).unsqueeze(1).repeat(1,self.num_prop, self.num_prop,1)# uncomment for geometric feats
        x_1 = self.image2rel_s(x_1)
        x_2 = self.image2rel_o(x_2)

        x_1 = x_1.unsqueeze(1).repeat(1, self.num_prop,1,1)
        x_2 = x_2.unsqueeze(2).repeat(1,1,self.num_prop,1)
        
        
        
        geo_feats = geo_feats.view(bs, self.num_prop, self.num_prop, -1)

        pair_score_geo = torch.cat([geo_feats, relation_geo], dim=-1)# uncomment for geometric feats
        pair_score_geo = self.cls_score_pair_geo(pair_score_geo.view(bs, self.num_prop*self.num_prop, -1))# uncomment for geometric feats
        pair_score_geo = pair_score_geo.view(bs, self.num_prop, self.num_prop)# uncomment for geometric feats
        pair_score_geo = F.sigmoid(pair_score_geo)# uncomment for geometric feats

        pair = torch.cat([x_1, x_2, geo_feats], dim=-1)
        pair = pair.view(bs, self.num_prop*self.num_prop, -1)
        pair = self.pair_proj(pair)
        pair = pair.view(bs, self.num_prop, self.num_prop, -1)
        # pair_score = torch.cat([x_1, x_2, relation, geo_feats], dim=-1)
        pair_score = torch.cat([pair, relation], dim=-1)
        pair_score = self.cls_pair_proj(pair_score.view(bs, self.num_prop*self.num_prop, -1))

        pair_score = self.cls_score_pair(pair_score)
        # pair_score = self.cls_score_pair_v2(pair_score.view(bs, self.num_prop*self.num_prop, -1))
        pair_score = pair_score.view(bs, self.num_prop, self.num_prop)
        pair_score = F.sigmoid(pair_score)
        

        if False:
            pair_score = pair_score*x_obj_test.unsqueeze(1)

            pair_score = pair_score*x_subj_test.unsqueeze(-1)
            # print(torch.max(pair_score))
            # pair_score = pair_score*x_obj.unsqueeze(1)

            # pair_score = pair_score*x_subj.unsqueeze(-1)

        x_subj_test = torch.clamp(x_subj_test, max = 0.99)
        x_subj_test = x_subj_test.view(-1) + 0.000001

        x_obj_test = torch.clamp(x_obj_test, max = 0.99)
        x_obj_test = x_obj_test.view(-1) + 0.000001


        x_subj = torch.cat([1-x_subj_test.unsqueeze(-1), x_subj_test.unsqueeze(-1)], dim =-1)
        x_obj = torch.cat([1-x_obj_test.unsqueeze(-1), x_obj_test.unsqueeze(-1)], dim =-1)

        x_subj = x_subj.view(-1, 2)
        x_obj = x_obj.view(-1,2)


        bbox_deltas = self.bbox_pred(x)
        return x_subj,x_obj,pair_score, bbox_deltas, [subj_indices, obj_indices]



class FastRCNNPredictorPaired(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictorPaired, self).__init__()
        in_channels = 1024
        # self.cls_score = nn.Linear(2*in_channels, num_classes)
        self.cls_score = nn.Linear(2*in_channels, 1)
        self.cls_score_pair = nn.Linear(3*in_channels, 1)
        self.bbox_pred = nn.Linear(in_channels, (num_classes+1) * 4)

        self.lmda = 0.1
        train = 128
        test = 300
        self.num_prop = test
        self.cos = CosineSimilarity(dim=-1)
        self.theta_rel = nn.Linear(in_channels, in_channels)
        self.epsilon = 1e-6
        
    def forward(self, x):
        subject = x[1]
        object = x[2]
        relation = x[3]
        x = x[0]
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        n_dim = 1024 # 256 before
        bs = relation.shape[0]
        relation = self.theta_rel(relation)

        
        # ''' # Message Passing in proposals
        
        x_1 = x.view(bs, self.num_prop, n_dim).unsqueeze(1).repeat(1, self.num_prop,1,1)
        x_2 = x.view(bs, self.num_prop, n_dim).unsqueeze(2).repeat(1,1,self.num_prop,1)

        relation = relation.unsqueeze(1).unsqueeze(1).repeat(1,self.num_prop, self.num_prop,1)

        pair_score = torch.cat([x_1, x_2, relation], dim=-1)
        pair_score = self.cls_score_pair(pair_score.view(bs, self.num_prop*self.num_prop, -1))
        pair_score = pair_score.view(bs, self.num_prop, self.num_prop)
        # x_2 = (relation.unsqueeze(1)*x_2).sum(-1)
        # x_1 = (relation.unsqueeze(1)*x_1).sum(-1)

        # pair_score = x_1.unsqueeze(-1) - x_2.unsqueeze(1) + self.epsilon
        # pair_score = torch.bmm(x_1, x_2.permute(0,2,1))
        pair_score = F.sigmoid(pair_score)
        
        # ''' # Message passing in queries
        # ob = self.linear_relation_query(relation)*self.linear_nodes_query(object)
        # sc = torch.bmm(self.linear_nodes_query(subject).unsqueeze(1), ob.unsqueeze(-1))
        
        # subject = subject + 0.1*sc.squeeze(1)*object
        # object = object + 0.1*sc.squeeze(1)*subject
        # ''' # Message passing in queries


        subject = subject.unsqueeze(1).repeat(1,self.num_prop,1)
        object = object.unsqueeze(1).repeat(1,self.num_prop,1)


        # x_subj = (F.cosine_similarity(x,subject) +1)/2
        # x_obj = (F.cosine_similarity(x, object) +1)/2
        x = x.view(-1, n_dim)
        subject = subject.view(-1, n_dim)
        object = object.view(-1, n_dim)

        
        x_subj = torch.cat([x,subject], dim=-1)
        x_subj = self.cls_score(x_subj)

        x_obj = torch.cat([x, object], dim=-1)
        x_obj = self.cls_score(x_obj)
        
        # x_subj = torch.bmm(x.unsqueeze(1),subject.unsqueeze(-1))
        # x_obj = torch.bmm(x.unsqueeze(1), object.unsqueeze(-1))

        x_subj = F.sigmoid(x_subj)
        x_obj = F.sigmoid(x_obj)

        # print(x_subj[0])
        # print(x_obj[0])


        # x_subj = self.cos(x,subject) #-----------------------------------------------------------|
        # x_obj = self.cos(x, object)                                                   #          |
             
                                                                                    ##               -> uncomment for cosine loss
        x_subj = x_subj.view(bs, self.num_prop)
        x_obj = x_obj.view(bs, self.num_prop)

        pair_score = pair_score*x_obj.unsqueeze(1)

        pair_score = pair_score*x_subj.unsqueeze(-1)

        
        '''
        x_intermediate_s = x_intermediate_s.sum(-1)

        x_subj_f = x_subj*x_intermediate_s
        

        x_intermediate_o = pair_score*x_subj.unsqueeze(-1)
        x_intermediate_o = x_intermediate_o.sum(1)

        x_obj_f = x_obj*x_intermediate_o
        '''
        x_subj = torch.cat([1-x_subj.unsqueeze(-1), x_subj.unsqueeze(-1)], dim =-1)
        x_obj = torch.cat([1-x_obj.unsqueeze(-1), x_obj.unsqueeze(-1)], dim =-1)

        x_subj = x_subj.view(-1, 2)
        x_obj = x_obj.view(-1,2)
        
        bbox_deltas = self.bbox_pred(x)

        return x_subj,x_obj, pair_score , bbox_deltas


class FastRCNNPredictorJoint(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes=2):
        super(FastRCNNPredictorJoint, self).__init__()
        num_classes = 2

        self.cls_score = nn.Linear(in_channels+256, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x, subject, object):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        subject = subject.unsqueeze(1).repeat(1,128,1)
        subject = subject.view(-1,256)
        object = object.unsqueeze(1).repeat(1,128,1)
        object = object.view(-1, 256)

        x_subj = self.cls_score(torch.cat([x, subject], dim=-1))
        x_obj = self.cls_score(torch.cat([x, object], dim=-1))


        # scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return x_subj,x_obj , bbox_deltas

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
    if True:
        print("Loading FPN model")
        # state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
        #                                       progress=progress)
        state_dict = torch.load("/path/to/fpn/model/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)                                      
        # model.load_state_dict(state_dict)
        overwrite_eps(model, 0.0)
    return model
