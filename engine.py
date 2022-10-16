import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn
from PIL import Image, ImageDraw
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils
import numpy as np
import torchvision.transforms.functional as F



def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        print(warmup_iters)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        print(lr_scheduler)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets ]

        if True:
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            # loss_dict_reduced = loss_dict
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types




@torch.no_grad()
def evaluateGNN(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    # model.roi_heads.box_predictor.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    recall_1_tp_seen = []
    recall_5_tp_seen = []
    recall_50_tp_seen = []
    

    recall_1_tp_unseen = []
    recall_5_tp_unseen = []
    recall_50_tp_unseen = []

    for image, targets in metric_logger.log_every(data_loader, 800, header):

        image = list(img.to(device) for img in image)
        tar_list = []
        for t in targets:
            tar = {}
            for k,v in t.items():
                try:
                    tar[k] = v.to(device)
                except:
                    tar[k] = v
            tar_list.append(tar)
        targets = tar_list
        if True:
            torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(image, targets)
            
            recall_1_tp_seen.extend(outputs[0])
            recall_5_tp_seen.extend(outputs[1])
            recall_50_tp_seen.extend(outputs[2])

            recall_1_tp_unseen.extend(outputs[3])
            recall_5_tp_unseen.extend(outputs[4])
            recall_50_tp_unseen.extend(outputs[5])

            img = inverse_normalize(tensor=image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

            img = F.to_pil_image(img[0].detach().cpu())
            img_1 = ImageDraw.Draw(img)

    print("Recall@1 seen:",sum(recall_1_tp_seen)/len(recall_1_tp_seen))
    print("Recall@5 seen:",sum(recall_5_tp_seen)/len(recall_5_tp_seen))
    print("Recall@50 seen:",sum(recall_50_tp_seen)/len(recall_50_tp_seen))

    print("Recall@1 unseen:",sum(recall_1_tp_unseen)/len(recall_1_tp_unseen))
    print("Recall@5 unseen:",sum(recall_5_tp_unseen)/len(recall_5_tp_unseen))
    print("Recall@50 unseen:",sum(recall_50_tp_unseen)/len(recall_50_tp_unseen))

    return None

# Part of code for generating visualizations is borrowed from https://github.com/ashkamath/mdetr 
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
import matplotlib.pyplot as plt
def plot_results(pil_img, scores, boxes, labels,id, masks=None):
    # plt.figure(figsize=(16,10))
    plt.figure()
    np_image = np.array(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if masks is None:
      masks = [None for _ in range(len(scores))]
    boxes = boxes.permute(1,0)
    assert len(scores) == len(boxes) == len(labels) == len(masks)
    for s, (xmin, ymin, xmax, ymax), l, mask, c in zip(scores, boxes.cpu().tolist(), labels, masks, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{l[0]}: {s:0.2f}'
        ax.text(xmin, ymin, text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        # print(text)
        if mask is None:
          continue
        np_image = apply_mask(np_image, mask, c)

        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
          # Subtract the padding and flip (y, x) to (x, y)
          verts = np.fliplr(verts) - 1
          p = Polygon(verts, facecolor="none", edgecolor=c)
          ax.add_patch(p)


    plt.imshow(np_image)
    plt.savefig('top_1_label_v2_unseen/'+str(id)+'_pred.png')
    plt.close()
@torch.no_grad()

def evaluateGNN_visualize(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    # model.roi_heads.box_predictor.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    recall_1_tp_seen = []
    recall_5_tp_seen = []
    recall_50_tp_seen = []
    

    recall_1_tp_unseen = []
    recall_5_tp_unseen = []
    recall_50_tp_unseen = []
    #print(len(data_loader))
    i = 0
    for image, targets in metric_logger.log_every(data_loader, 800, header):
        image = list(img.to(device) for img in image)
        tar_list = []
        for t in targets:
            tar = {}
            for k,v in t.items():
                try:
                    tar[k] = v.to(device)
                except:
                    tar[k] = v
            tar_list.append(tar)
        targets = tar_list
        if True:
            torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(image, targets)
            
            pred_boxes = outputs[6]
            prob_box = outputs[7]
            box_names = targets[0]['node_names']
            recall_1_tp_seen.extend(outputs[0])
            recall_5_tp_seen.extend(outputs[1])
            recall_50_tp_seen.extend(outputs[2])

            recall_1_tp_unseen.extend(outputs[3])
            recall_5_tp_unseen.extend(outputs[4])
            recall_50_tp_unseen.extend(outputs[5])

            # print(pred_boxes)
            img = inverse_normalize(tensor=image[0], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            img = [img]
            img = F.to_pil_image(img[0].detach().cpu())
            img.save("top_1_label_v2_unseen/"+str(i)+'_'+targets[0]['name']+'_original.jpg')
            img_1 = ImageDraw.Draw(img)
            
            plot_results(img, prob_box, pred_boxes, box_names,i)
            color = "red"
            i +=1

    print("Recall@1 seen:",sum(recall_1_tp_seen)/len(recall_1_tp_seen))
    print("Recall@5 seen:",sum(recall_5_tp_seen)/len(recall_5_tp_seen))
    print("Recall@50 seen:",sum(recall_50_tp_seen)/len(recall_50_tp_seen))

    print("Recall@1 unseen:",sum(recall_1_tp_unseen)/len(recall_1_tp_unseen))
    print("Recall@5 unseen:",sum(recall_5_tp_unseen)/len(recall_5_tp_unseen))
    print("Recall@50 unseen:",sum(recall_50_tp_unseen)/len(recall_50_tp_unseen))

    return None


def inverse_normalize(tensor, mean, std):
    
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
