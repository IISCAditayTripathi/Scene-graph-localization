
from math import gamma
import torch
from torch.autograd.grad_mode import enable_grad
import torchvision

import torch.nn.functional as F
from torch import nn, Tensor

from torchvision.ops import boxes as box_ops

from torchvision.ops import roi_align

from . import _utils as det_utils

from typing import Optional, List, Dict, Tuple
from collections import defaultdict


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = det_utils.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss
def proposal_pair_loss(proposal_pair_scores, labels):
    labels = torch.cat(labels, dim=0)
    subject_labels = labels.clone()
    object_labels = labels.clone()
    bs,_,_ = proposal_pair_scores.shape
    gamma = 1

    subject_labels[subject_labels==2] = 0
    object_labels[object_labels==1] = 0

    object_labels[object_labels==2] = 1

    labels = torch.bmm(subject_labels.view(bs, 128,1).float(), object_labels.view(bs,1,128).float())

    labels = labels.view(-1).long()
    
    # proposal_pair_scores = torch.nn.functional.sigmoid(proposal_pair_scores)
    proposal_pair_scores = proposal_pair_scores.view(-1)

    proposal_pair_scores = torch.cat([1-proposal_pair_scores.unsqueeze(-1), proposal_pair_scores.unsqueeze(-1)], dim=-1)
    
    pair_loss = torch.nn.functional.nll_loss(torch.log(proposal_pair_scores), labels, reduction='none')
    
    # pt = torch.exp(-pair_loss)
    
    # pair_loss = -((1-pt)**gamma)*(-pair_loss)
    
    pair_loss = pair_loss.mean()
    return pair_loss

def proposal_pair_loss_sorted(proposal_pair_scores, labels, selected_labels):
    subj_indices = selected_labels[0]
    obj_indices = selected_labels[1]
    labe = []
    for label in labels:
        labe.append(label.unsqueeze(0))

    labels = torch.cat(labe, dim=0)

    # labels = torch.cat(labels.unsqueeze(0), dim=0)
    subject_labels = labels.clone()
    object_labels = labels.clone()

    bs = labels.shape[0]
    subject_new = []
    object_new = []
    for i in range(bs):
            subject_new.append(subject_labels[i, subj_indices[i]].unsqueeze(0))
            object_new.append(object_labels[i, obj_indices[i]].unsqueeze(0))

    subject_labels = torch.cat(subject_new, dim=0)
    object_labels = torch.cat(object_new, dim=0)


    subject_labels = subject_labels.view(-1)
    object_labels = object_labels.view(-1)

    bs,_,_ = proposal_pair_scores.shape
    gamma = 0

    subject_labels[subject_labels==2] = 0
    object_labels[object_labels==1] = 0

    object_labels[object_labels==2] = 1
    num_prop = 32
    labels = torch.bmm(subject_labels.view(bs, num_prop,1).float(), object_labels.view(bs,1,num_prop).float())

    labels = labels.view(-1).long()
    
    # proposal_pair_scores = torch.nn.functional.sigmoid(proposal_pair_scores)
    proposal_pair_scores = proposal_pair_scores.view(-1)

    proposal_pair_scores = torch.cat([1-proposal_pair_scores.unsqueeze(-1), proposal_pair_scores.unsqueeze(-1)], dim=-1)
    
    pair_loss = torch.nn.functional.nll_loss(torch.log(proposal_pair_scores), labels, reduction='none')
    
    pt = torch.exp(-pair_loss)
    
    pair_loss = -((1-pt)**gamma)*(-pair_loss)
    
    pair_loss = pair_loss.mean()
    return pair_loss

def proposal_pair_loss_sorted_full(proposal_pair_scores, labels, selected_labels):
    subj_indices = selected_labels[0]
    obj_indices = selected_labels[1]
    labe = []
    for label in labels:
        labe.append(label.unsqueeze(0))

    labels = torch.cat(labe, dim=0)

    subject_labels = labels.clone()
    object_labels = labels.clone()

    bs = labels.shape[0]
    subject_new = []
    object_new = []
    for i in range(bs):
            subject_new.append(subject_labels[i, subj_indices[i]].unsqueeze(0))
            object_new.append(object_labels[i, obj_indices[i]].unsqueeze(0))

    subject_labels = torch.cat(subject_new, dim=0)
    object_labels = torch.cat(object_new, dim=0)

    subject_labels = subject_labels.view(-1)
    object_labels = object_labels.view(-1)

    bs,_,_ = proposal_pair_scores.shape
    gamma = 0

    
    
    subject_labels[subject_labels%2==0] = 0
    object_labels = object_labels - subject_labels

    ss = subject_labels*(subject_labels+1)
    
    ss = ss[ss>0]
    ss = torch.unique(ss)
    
    num_prop = 48
    
    labels = torch.bmm(subject_labels.view(bs, num_prop,1).float(), object_labels.view(bs,1,num_prop).float())
    # print(torch.where(labels==2))
    for ele in ss:
        labels[labels==ele] = 1
    
    labels[labels !=1] = 0
    # print(labels[0])
    proposal_pair_scores = torch.clamp(proposal_pair_scores, max = 0.99)
    proposal_pair_scores = proposal_pair_scores.view(-1) + 0.00000001
    # print(torch.sum(labels))
    proposal_pair_scores_bs = []
    labels_bs = []
    # print(labels.shape)
    # print(proposal_pair_scores.shape)
    labels = labels.view(bs, -1)
    proposal_pair_scores = proposal_pair_scores.view(bs, -1)
    
        
    labels = labels.view(-1).long()
    proposal_pair_scores = proposal_pair_scores.view(-1)
    
    l1_norm = torch.norm(proposal_pair_scores, p=1)
    l1_norm = 0.1*l1_norm
    # print(labels.shape)
    a = labels*proposal_pair_scores
    a = a[a>0]
    b = (1-labels)*proposal_pair_scores
    b = b[b>0]
    


    proposal_pair_scores = torch.cat([1-proposal_pair_scores.unsqueeze(-1), proposal_pair_scores.unsqueeze(-1)], dim=-1)
    
    pair_loss = torch.nn.functional.nll_loss(torch.log(proposal_pair_scores), labels, reduction='none')
    
    pt = torch.exp(-pair_loss)
    # gamma = 0.8
    gamma = 0
    pair_loss = -((1-pt)**gamma)*(-pair_loss)
    
    pair_loss = pair_loss.mean()
    return pair_loss, l1_norm

def proposal_pair_loss_sorted_full_gm(proposal_pair_scores, labels, selected_labels):
    subj_indices = selected_labels[0]
    obj_indices = selected_labels[1]
    
    labels = labels.unsqueeze(0).repeat(subj_indices.shape[0],1)
    
    subject_labels = labels.clone()
    object_labels = labels.clone()

    bs = labels.shape[0]
    subject_new = []
    object_new = []
    for i in range(bs):
            subject_new.append(subject_labels[i, subj_indices[i]].unsqueeze(0))
            object_new.append(object_labels[i, obj_indices[i]].unsqueeze(0))

    subject_labels = torch.cat(subject_new, dim=0)
    object_labels = torch.cat(object_new, dim=0)
    
    subj_indices = []
    obj_indices = []
    for i in range(subject_labels.shape[0]):
        subj_indices.append(2*i + 1)
        obj_indices.append(2*i + 2)
    
    subj_indices = torch.tensor(subj_indices).cuda()
    obj_indices = torch.tensor(obj_indices).cuda()
    
    subject_labels = subject_labels/subj_indices.unsqueeze(1)
    object_labels = object_labels/obj_indices.unsqueeze(1)
    
    subject_labels[subject_labels!=1] = 0
    object_labels[object_labels!=1] = 0
    subject_labels = subject_labels.long()
    object_labels = object_labels.long()
    
    num_prop = 48
    labels = torch.bmm(subject_labels.view(bs, num_prop,1).float(), object_labels.view(bs,1,num_prop).float())

    labels = labels.view(-1)
    proposal_pair_scores = torch.clamp(proposal_pair_scores, max = 0.99)
    proposal_pair_scores = proposal_pair_scores.view(-1) + 0.00000001
    # print(torch.sum(labels))
    proposal_pair_scores_bs = []
    labels_bs = []
    labels = labels.view(bs, -1)
    proposal_pair_scores = proposal_pair_scores.view(bs, -1)    
        
    labels = labels.view(-1).long()
    proposal_pair_scores = proposal_pair_scores.view(-1)
    
    l1_norm = torch.norm(proposal_pair_scores, p=1)
    l1_norm = 0.1*l1_norm
    # print(labels.shape)
    a = labels*proposal_pair_scores
    a = a[a>0]
    b = (1-labels)*proposal_pair_scores
    b = b[b>0]
    
    proposal_pair_scores = torch.cat([1-proposal_pair_scores.unsqueeze(-1), proposal_pair_scores.unsqueeze(-1)], dim=-1)
    
    pair_loss = torch.nn.functional.nll_loss(torch.log(proposal_pair_scores), labels, reduction='none')
    
    pt = torch.exp(-pair_loss)
    # gamma = 0.8
    gamma = 0
    pair_loss = -((1-pt)**gamma)*(-pair_loss)
    
    pair_loss = pair_loss.mean()
    return pair_loss, _

def proposal_pair_loss_sorted_full_gm_nodes(proposal_pair_scores, labels, selected_labels):
    subj_indices = selected_labels[0]
    obj_indices = selected_labels[1]

    
    subject_labels = labels[0].unsqueeze(0).repeat(subj_indices.shape[0],1)
    object_labels = labels[1].unsqueeze(0).repeat(subj_indices.shape[0],1)

    bs = subject_labels.shape[0]
    subject_new = []
    object_new = []
    for i in range(bs):
            subject_new.append(subject_labels[i, subj_indices[i]].unsqueeze(0))
            object_new.append(object_labels[i, obj_indices[i]].unsqueeze(0))

    subject_labels = torch.cat(subject_new, dim=0)
    object_labels = torch.cat(object_new, dim=0)
    
    subj_indices = []
    obj_indices = []
    for i in range(subject_labels.shape[0]):
        subj_indices.append(2*i + 1)
        obj_indices.append(2*i + 2)
    
    subj_indices = torch.tensor(subj_indices).cuda()
    obj_indices = torch.tensor(obj_indices).cuda()
    
    subject_labels = subject_labels/subj_indices.unsqueeze(1)
    object_labels = object_labels/obj_indices.unsqueeze(1)
    
    subject_labels[subject_labels!=1] = 0
    object_labels[object_labels!=1] = 0
    subject_labels = subject_labels.long()
    object_labels = object_labels.long()
    
    
    num_prop = 48
    labels = torch.bmm(subject_labels.view(bs, num_prop,1).float(), object_labels.view(bs,1,num_prop).float())
    
    labels = labels.view(-1)
    proposal_pair_scores = torch.clamp(proposal_pair_scores, max = 0.99)
    proposal_pair_scores = proposal_pair_scores.view(-1) + 0.00000001
    
    proposal_pair_scores_bs = []
    labels_bs = []
    
    labels = labels.view(bs, -1)
    proposal_pair_scores = proposal_pair_scores.view(bs, -1)    
        
    labels = labels.view(-1).long()
    proposal_pair_scores = proposal_pair_scores.view(-1)
    
    
    
    a = labels*proposal_pair_scores
    a = a[a>0]
    b = (1-labels)*proposal_pair_scores
    b = b[b>0]
    
    proposal_pair_scores = torch.cat([1-proposal_pair_scores.unsqueeze(-1), proposal_pair_scores.unsqueeze(-1)], dim=-1)
    

    pair_loss = torch.nn.functional.nll_loss(torch.log(proposal_pair_scores), labels, reduction='none')
    
    pt = torch.exp(-pair_loss)
    
    gamma = 0
    pair_loss = -((1-pt)**gamma)*(-pair_loss)
    
    pair_loss = pair_loss.mean()
    return pair_loss




def fastrcnn_loss_joint(class_logits_subject, class_logits_obj, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor,Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
    # bs = relation_score.shape[0]

    labels = torch.cat(labels, dim=0)
    
    regression_targets = torch.cat(regression_targets, dim=0)

    subject_labels = labels.clone()
    object_labels = labels.clone()
    box_labels = labels.clone()
    
    # box_labels[box_labels==2] = 1 # may need to uncomment
    
    subject_labels[subject_labels==2] = 0
    object_labels[object_labels==1] = 0
    object_labels[object_labels==2] = 1
    # labels_subj = labels[labels==2]

    # print(subject_labels.sum())
    # print(object_labels.sum())


    # classification_loss_subj = F.cross_entropy(class_logits_subject, subject_labels) # uncomment for CE
    # classification_loss_obj = F.cross_entropy(class_logits_obj, object_labels)       # Uncomment for CE
    gamma = 0
    classification_loss_subj = F.nll_loss(torch.log(class_logits_subject), subject_labels, reduction='none') # uncomment for Cos
    pt = torch.exp(-classification_loss_subj)
    classification_loss_subj = -((1-pt)**gamma)*(-classification_loss_subj)
    
    classification_loss_subj = classification_loss_subj.mean()


    classification_loss_obj = F.nll_loss(torch.log(class_logits_obj), object_labels, reduction='none')       # uncomment for Cos
    pt = torch.exp(-classification_loss_obj)
    classification_loss_obj = -((1-pt)**gamma)*(-classification_loss_obj)
    classification_loss_obj = classification_loss_obj.mean()

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(box_labels > 0)[0]
    labels_pos = box_labels[sampled_pos_inds_subset]
    N, num_classes = class_logits_subject.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    

    box_loss = det_utils.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    )
    box_loss = box_loss / labels.numel()
    

    return classification_loss_subj, classification_loss_obj, box_loss, subject_labels, object_labels

def box_loss(class_logits_subject, class_logits_obj, box_regression, labels, regression_targets):
    # labels = torch.cat(labels, dim=0)
    # regression_targets = torch.cat(regression_targets, dim=0)
    subject_labels = labels.clone()
    object_labels = labels.clone()
    box_labels = labels.clone()
    box_labels[box_labels != 0]= 1
    
    

    subject_labels[subject_labels%2==0] = 0
    subject_labels[subject_labels%2!=0] = 1
    object_labels[object_labels%2!=0] = 0
    
    object_labels[object_labels!=0] = 1

    oj = object_labels.clone()
    oj[oj!=0] = oj[oj !=0] +1
    # box_labels = subject_labels.clone() + oj

    sampled_pos_inds_subset = torch.where(box_labels > 0)[0]
    labels_pos = box_labels[sampled_pos_inds_subset]
    # N, num_classes = class_logits_subject.shape
    N = box_regression.shape[0]
    
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    

    box_loss = det_utils.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    )
    box_loss = box_loss / labels.numel()

    return box_loss




def fastrcnn_loss_joint_full(class_logits_subject, class_logits_obj, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor,Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
    # bs = relation_score.shape[0]
    

    labels = torch.cat(labels, dim=0)
    
    
    regression_targets = torch.cat(regression_targets, dim=0)

    subject_labels = labels.clone()
    object_labels = labels.clone()
    box_labels = labels.clone()
    
    
    
    subject_labels[subject_labels%2==0] = 0
    subject_labels[subject_labels%2!=0] = 1
    object_labels[object_labels%2!=0] = 0
    
    object_labels[object_labels!=0] = 1

    oj = object_labels.clone()
    oj[oj!=0] = oj[oj !=0] +1
    box_labels = subject_labels.clone() + oj
    
    class_log_subj = class_logits_subject[:,1]
    class_log_obj = class_logits_obj[:,1]
    l1_subj = torch.norm(class_log_subj, p=1)
    l1_obj = torch.norm(class_log_obj, p=1)
    

    gamma = 0
    classification_loss_subj = F.nll_loss(torch.log(class_logits_subject), subject_labels, reduction='none') # uncomment for Cos
    pt = torch.exp(-classification_loss_subj)
    classification_loss_subj = -((1-pt)**gamma)*(-classification_loss_subj)
    
    classification_loss_subj = classification_loss_subj.mean()


    classification_loss_obj = F.nll_loss(torch.log(class_logits_obj), object_labels, reduction='none')       # uncomment for Cos
    pt = torch.exp(-classification_loss_obj)
    classification_loss_obj = -((1-pt)**gamma)*(-classification_loss_obj)
    classification_loss_obj = classification_loss_obj.mean()

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(box_labels > 0)[0]
    labels_pos = box_labels[sampled_pos_inds_subset]
    N, num_classes = class_logits_subject.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    

    box_loss = det_utils.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    )
    box_loss = box_loss / labels.numel()
    

    return classification_loss_subj, classification_loss_obj, box_loss, subject_labels, object_labels, 0.1*l1_subj, 0.1*l1_obj

def fastrcnn_loss_joint_full_gm(class_logits_subject, class_logits_obj, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor,Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """


    subject_labels = labels.clone()
    object_labels = labels.clone()
    box_labels = labels.clone()
    
    
    
    subject_labels[subject_labels%2==0] = 0
    object_labels[object_labels%2!=0] = 0
    
    
    subject_labels = subject_labels.unsqueeze(0).repeat(class_logits_subject.shape[0], 1)
    object_labels = object_labels.unsqueeze(0).repeat(class_logits_subject.shape[0], 1)
    subj_indices = []
    obj_indices = []
    for i in range(subject_labels.shape[0]):
        subj_indices.append(2*i + 1)
        obj_indices.append(2*i + 2)
    subj_indices = torch.tensor(subj_indices)
    obj_indices = torch.tensor(obj_indices)
    
    subject_labels = subject_labels/subj_indices.unsqueeze(1)
    object_labels = object_labels/obj_indices.unsqueeze(1)
    
    subject_labels[subject_labels!=1] = 0
    object_labels[object_labels!=1] = 0
    subject_labels = subject_labels.long()
    object_labels = object_labels.long()
    

    oj = object_labels.clone()
    oj[oj!=0] = oj[oj !=0] +1
    box_labels = subject_labels.clone() + oj
    class_logits_subject = class_logits_subject.view(-1,2)
    class_logits_obj = class_logits_obj.view(-1,2)
    subject_labels = subject_labels.view(-1)
    object_labels = object_labels.view(-1)

    class_log_subj = class_logits_subject[:,1]
    class_log_obj = class_logits_obj[:,1]
    

    gamma = 0
    
    classification_loss_subj = F.nll_loss(torch.log(class_logits_subject), subject_labels, reduction='none') # uncomment for Cos
    pt = torch.exp(-classification_loss_subj)
    classification_loss_subj = -((1-pt)**gamma)*(-classification_loss_subj)
    
    classification_loss_subj = classification_loss_subj.mean()


    classification_loss_obj = F.nll_loss(torch.log(class_logits_obj), object_labels, reduction='none')       # uncomment for Cos
    pt = torch.exp(-classification_loss_obj)
    classification_loss_obj = -((1-pt)**gamma)*(-classification_loss_obj)
    classification_loss_obj = classification_loss_obj.mean()
    
    

    return classification_loss_subj, classification_loss_obj

def fastrcnn_loss_joint_full_gm_nodes(node_logits, labels):
    # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    node_labels = labels.clone()
    
    
    node_labels = node_labels.unsqueeze(0).repeat(node_logits.shape[0], 1)
    node_indices = []
    for i in range(node_labels.shape[0]):
        node_indices.append(i+1)
    
    node_indices = torch.tensor(node_indices).cuda()
    
    node_labels = node_labels/node_indices.unsqueeze(1)
    
    node_labels[node_labels!=1] = 0
    node_labels = node_labels.long()
    

    class_logits_node = node_logits.view(-1,2)
    
    node_labels = node_labels.view(-1)
    

    class_log_subj = class_logits_node[:,1]
    if True:
        score_prob_node = class_logits_node[:,1]
        bs = score_prob_node.shape[0]//256

        score_label_node = node_labels.view(bs,-1).float()
        
        gt_map_node = torch.abs(score_label_node.unsqueeze(1)-score_label_node.unsqueeze(-1))
        
        # score_prob_subject = F.softmax(class_logits_subject,1)[:,1] # uncomment for CE
        
        
                
        score_prob_node = score_prob_node.view(bs, -1)
        # print(score_prob_subject[0])
        pr_map = torch.abs(score_prob_node.unsqueeze(1)-score_prob_node.unsqueeze(-1))
        target = -((gt_map_node-1)**2) + gt_map_node
        
        margin_loss_node = 3 * torch.nn.functional.margin_ranking_loss(pr_map, gt_map_node, target, margin=-0.3)
    

    gamma = 0

    classification_loss_node = F.nll_loss(torch.log(class_logits_node), node_labels, reduction='none') # uncomment for Cos
    pt = torch.exp(-classification_loss_node)
    classification_loss_node = -((1-pt)**gamma)*(-classification_loss_node)
    
    classification_loss_node = classification_loss_node.mean()



    return classification_loss_node, margin_loss_node




def fastrcnn_loss_joint_sorted(class_logits_subject, class_logits_obj, box_regression, labels, regression_targets, selected_labels):

    subj_indices = selected_labels[0]
    obj_indices = selected_labels[1]
    labe = []
    for label in labels:
        labe.append(label.unsqueeze(0))

    labels = torch.cat(labe, dim=0)
    
    regression_targets = torch.cat(regression_targets, dim=0)

    subject_labels = labels.clone()
    object_labels = labels.clone()
    box_labels = labels.clone()
    bs = labels.shape[0]
    subject_new = []
    object_new = []
    for i in range(bs):
            subject_new.append(subject_labels[i, subj_indices[i], :].unsqueeze(0))
            object_new.append(object_labels[i, obj_indices[i], :].unsqueeze(0))

    subject_labels = torch.cat(subject_new, dim=0)
    object_labels = torch.cat(object_new, dim=0)

    
    subject_labels = subject_labels.view(-1)
    object_labels = object_labels.view(-1)
    
    subject_labels[subject_labels==2] = 0
    object_labels[object_labels==1] = 0
    object_labels[object_labels==2] = 1
    gamma = 1
    classification_loss_subj = F.nll_loss(torch.log(class_logits_subject), subject_labels, reduction='none') # uncomment for Cos
    pt = torch.exp(-classification_loss_subj)
    classification_loss_subj = -((1-pt)**gamma)*(-classification_loss_subj)
    classification_loss_subj = classification_loss_subj.mean()


    classification_loss_obj = F.nll_loss(torch.log(class_logits_obj), object_labels, reduction='none')       # uncomment for Cos
    pt = torch.exp(-classification_loss_obj)
    classification_loss_obj = -((1-pt)**gamma)*(-classification_loss_obj)
    classification_loss_obj = classification_loss_obj.mean()

    sampled_pos_inds_subset = torch.where(box_labels > 0)[0]
    labels_pos = box_labels[sampled_pos_inds_subset]
    N, num_classes = class_logits_subject.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    

    box_loss = det_utils.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    )
    box_loss = box_loss / labels.numel()
    

    return classification_loss_subj, classification_loss_obj, box_loss, subject_labels, object_labels

def fastrcnn_loss_margin(class_logits_subject, class_logits_obj, triplet_loss, subject_labels, object_labels, bs):
    # type: (Tensor, Tensor,Tensor,Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
    # print(subject_labels.sum()/6)
    score_label_subj = subject_labels.view(bs, -1).float()
    gt_map_subj = torch.abs(score_label_subj.unsqueeze(1)-score_label_subj.unsqueeze(-1))
    # score_prob_subject = F.softmax(class_logits_subject,1)[:,1] # uncomment for CE
    score_prob_subject = class_logits_subject[:,1]
    
            
    score_prob_subject = score_prob_subject.view(bs, -1)
    # print(score_prob_subject[0])
    pr_map = torch.abs(score_prob_subject.unsqueeze(1)-score_prob_subject.unsqueeze(-1))
    target = -((gt_map_subj-1)**2) + gt_map_subj
    
    margin_loss_subj = 2 * triplet_loss(pr_map, gt_map_subj, target)


    score_label_obj = object_labels.view(bs, -1).float()
    gt_map_obj = torch.abs(score_label_obj.unsqueeze(1)-score_label_obj.unsqueeze(-1))
    # score_prob_obj = F.softmax(class_logits_obj,1)[:,1]  #uncomment for CE
    score_prob_obj = class_logits_obj[:,1]
       
    score_prob_obj = score_prob_obj.view(bs, -1)
    pr_map = torch.abs(score_prob_obj.unsqueeze(1)-score_prob_obj.unsqueeze(-1))
    target = -((gt_map_obj-1)**2) + gt_map_obj
    margin_loss_obj = 2 * triplet_loss(pr_map, gt_map_obj, target)
    
    return margin_loss_subj, margin_loss_obj
