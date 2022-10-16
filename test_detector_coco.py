#!/usr/bin/env python
# coding: utf-8



from multiprocessing import Value
import sys
sys.executable



import random
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import utils
import torch
import torchvision
import torch.utils.data as data
import json, os, random, math
from collections import defaultdict
import numpy as np
import pycocotools.mask as mask_utils
from skimage.transform import resize as imresize

from PIL import Image, ImageDraw
from lib.faster_rcnn import  FastRCNNPredictorPairedSortedGNNFull
from torchvision.transforms import functional as F
from engine import  evaluateGNN
import lib
import gensim
from gensim import downloader
import torch.multiprocessing
from collections import defaultdict
torch.multiprocessing.set_sharing_strategy('file_system')
torch.multiprocessing.set_sharing_strategy('file_system')
glove_vectors = downloader.load('glove-wiki-gigaword-300')



print (torch.cuda.is_available())
print (torch.cuda.device_count())
print (torch.cuda.get_device_name())
device = torch.device("cuda")


import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    
    return T.Compose(transforms)



def seg_to_mask(seg, width=1.0, height=1.0):
  """
  Tiny utility for decoding segmentation masks using the pycocotools API.
  """
  if type(seg) == list:
    rles = mask_utils.frPyObjects(seg, height, width)
    rle = mask_utils.merge(rles)
  elif type(seg['counts']) == list:
    rle = mask_utils.frPyObjects(seg, height, width)
  else:
    rle = seg
  return mask_utils.decode(rle)

class COCODataset(data.Dataset):
    def __init__(self,glove_vectors, transforms=None):

        self.transforms = transforms
        self.image_dir = "/path/to/coco/val2017"
        self.include_relationships = True
        instances_json = "/path/to/coco/coco/instances_val2017.json"
        with open(instances_json, 'r') as f:
            instances_data = json.load(f)

        stuff_json = "/path/to/coco/stuff_val2017.json"
        with open(stuff_json, 'r') as f:
            stuff_data = json.load(f)

        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        for image_data in instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)
        
        self.vocab = {
                    'object_name_to_idx': {},
                    'pred_name_to_idx': {},
                    }

        object_idx_to_name = {}
        all_instance_categories = []

        for category_data in instances_data['categories']:
            category_id = category_data['id']
            category_name = category_data['name']
            all_instance_categories.append(category_name)
            object_idx_to_name[category_id] = category_name
            self.vocab['object_name_to_idx'][category_name] = category_id
            all_stuff_categories = []
        if stuff_data:
            for category_data in stuff_data['categories']:
                category_name = category_data['name']
                category_id = category_data['id']
                all_stuff_categories.append(category_name)
                object_idx_to_name[category_id] = category_name
                self.vocab['object_name_to_idx'][category_name] = category_id
        self.image_id_to_objects = defaultdict(list)
        instance_whitelist = None
        if instance_whitelist is None:
            instance_whitelist = all_instance_categories
        stuff_whitelist = None
        if stuff_whitelist is None:
            stuff_whitelist = all_stuff_categories
            category_whitelist = set(instance_whitelist) | set(stuff_whitelist)
        min_object_size = 0.02
        include_other = False
        for object_data in instances_data['annotations']:
            image_id = object_data['image_id']
            _, _, w, h = object_data['bbox']
            W, H = self.image_id_to_size[image_id]
            box_area = (w * h) / (W * H)
            box_ok = box_area > min_object_size
            object_name = object_idx_to_name[object_data['category_id']]
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other' or include_other
            if box_ok and category_ok and other_ok:
                self.image_id_to_objects[image_id].append(object_data)
        stuff_only = True
        if stuff_data:
            image_ids_with_stuff = set()
            for object_data in stuff_data['annotations']:
                image_id = object_data['image_id']
                image_ids_with_stuff.add(image_id)
                _, _, w, h = object_data['bbox']
                W, H = self.image_id_to_size[image_id]
                box_area = (w * h) / (W * H)
                box_ok = box_area > min_object_size
                object_name = object_idx_to_name[object_data['category_id']]
                category_ok = object_name in category_whitelist
                other_ok = object_name != 'other' or include_other
                if box_ok and category_ok and other_ok:
                    self.image_id_to_objects[image_id].append(object_data)
            if stuff_only:
                new_image_ids = []
                for image_id in self.image_ids:
                    if image_id in image_ids_with_stuff:
                        new_image_ids.append(image_id)
                    self.image_ids = new_image_ids

                all_image_ids = set(self.image_id_to_filename.keys())
                image_ids_to_remove = all_image_ids - image_ids_with_stuff
                for image_id in image_ids_to_remove:
                    self.image_id_to_filename.pop(image_id, None)
                    self.image_id_to_size.pop(image_id, None)
                    self.image_id_to_objects.pop(image_id, None)
        self.data = {}


        name_to_idx = self.vocab['object_name_to_idx']
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx_to_name = ['NONE'] * (1 + max_object_idx)

        for name, idx in self.vocab['object_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['object_idx_to_name'] = idx_to_name
        new_image_ids = []
        total_objs = 0
        min_objects_per_image = 3  #3
        
        max_objects_per_image = 8 #8

        for image_id in self.image_ids:
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
            if min_objects_per_image <= num_objs <= max_objects_per_image:
                new_image_ids.append(image_id)
        self.image_ids = new_image_ids
        random.seed(14)

        self.vocab['pred_idx_to_name'] = [
                                        'left of',
                                        'right of',
                                        'above',
                                        'below',
                                        'inside',
                                        'surrounding',
                                        ]
        self.vocab['pred_name_to_idx'] = {}

        for idx, name in enumerate(self.vocab['pred_idx_to_name']):
            self.vocab['pred_name_to_idx'][name] = idx


        self.w2v = glove_vectors

    def load_image(self, index):
        image_path = os.path.join(self.image_dir, "%d.jpg"%(index))
        if os.path.exists(image_path):
            img = Image.open(image_path)
        else:
            image_path = os.path.join(self.image_dir+"_2", "%d.jpg"%(index))
            img = Image.open(image_path)

        return img
            
        
    def __getitem__(self, index):
        

        image_id = self.image_ids[index]
        
        filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, filename)
        img = Image.open(image_path)
        WW, HH = img.size

        objs, boxes, masks = [], [], []
        for object_data in self.image_id_to_objects[image_id]:
            objs.append(object_data['category_id'])
            x, y, w, h = object_data['bbox']
            boxes.append(torch.FloatTensor([x, y, x+w, y+w]))
            
            mask = seg_to_mask(object_data['segmentation'], WW, HH)
            
            mx0, mx1 = int(round(x)), int(round(x + w))
            my0, my1 = int(round(y)), int(round(y + h))
            mx1 = max(mx0 + 1, mx1)
            my1 = max(my0 + 1, my1)
            
            mask = mask[my0:my1, mx0:mx1]
            self.mask_size = WW//4
            mask = imresize(255.0 * mask, (self.mask_size, self.mask_size),
                      mode='constant')
            mask = torch.from_numpy((mask > 128).astype(np.int64))
            
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
        objs = torch.LongTensor(objs)
        boxes = torch.stack(boxes, dim=0)
        obj_centers = []
        _, MH, MW = masks.size()
        for i, obj_idx in enumerate(objs):
            x0, y0, x1, y1 = boxes[i]
            mask = (masks[i] == 1)
            xs = torch.linspace(x0, x1, MW).view(1, MW).expand(MH, MW)
            ys = torch.linspace(y0, y1, MH).view(MH, 1).expand(MH, MW)
            if mask.sum() == 0:
                mean_x = 0.5 * (x0 + x1)
                mean_y = 0.5 * (y0 + y1)
            else:
                mean_x = xs[mask].mean()
                mean_y = ys[mask].mean()
            obj_centers.append([mean_x, mean_y])
        obj_centers = torch.FloatTensor(obj_centers)
        triples = []
        num_objs = objs.size(0)
        real_objs = []
        __image__ = "__image__"
        real_objs = objs
        objs_ids = [i for i in range(len(real_objs))]
        rel_happened = []
        unique_rel = []
        rels = []
        i = 0
        s_o_occured = []
        rel_idx2id = {}
        
        for cur, name_id in enumerate(real_objs):
            choices = [obj for obj in objs_ids if obj != cur]
            if len(choices) == 0 or not self.include_relationships:
                break
            other = random.choice(choices)
            if random.random() > 0.5:
                s, o = cur, other
            else:
                s, o = other, cur

            if str(s)+'_'+str(o) in s_o_occured:
                s,o = o,s
                if str(s)+'_'+str(o) in s_o_occured:
                    continue


            sx0, sy0, sx1, sy1 = boxes[s]
            ox0, oy0, ox1, oy1 = boxes[o]
            d = obj_centers[s] - obj_centers[o]
            theta = math.atan2(d[1], d[0])
            if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
                p = 'surrounding'
            elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
                p = 'inside'
            elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
                p = 'left of'
            elif -3 * math.pi / 4 <= theta < -math.pi / 4:
                p = 'above'
            elif -math.pi / 4 <= theta < math.pi / 4:
                p = 'right of'
            elif math.pi / 4 <= theta < 3 * math.pi / 4:
                p = 'below'
            
            rel_idx = self.vocab['pred_name_to_idx'][p]
            
            rel_name = p.encode().decode("utf-8","ignore").split(' ')
            
            if len(rel_name)>1:
                rel_name = torch.as_tensor([self.w2v[str(r)] for r in rel_name], dtype=torch.float32).mean(0)
            else:
                rel_name = torch.as_tensor(self.w2v[str(rel_name[0])], dtype=torch.float32)
            rels.append(rel_name.unsqueeze(0))
            if rel_idx in rel_happened:
                p = rel_idx2id[rel_idx]
            else:
                unique_rel.append(rel_name.unsqueeze(0))
                rel_happened.append(rel_idx)
                rel_idx2id[rel_idx] = i
                p = i
                i = i+1
                

        

            triples.append([int(s), int(p), int(o)])
            s_o_occured.append(str(s)+'_'+str(o))
        
        unique_nodes = []
        for cur, name_id in enumerate(real_objs):
            id_name = self.vocab['object_idx_to_name'][name_id]
            # print(id_name)
            if id_name.encode().decode() == "playingfield":
                id_name = "playing field"

            if id_name.encode().decode() == "waterdrops":
                id_name = "water drops"
            id_name = id_name.encode().decode("utf-8","ignore").replace('-',' ').split(" ")
            # print(id_name)
            if len(id_name)>1:
                
                id_name = torch.as_tensor([self.w2v[str(r)] for r in id_name], dtype=torch.float32).mean(0)
                
            else:
                
                id_name = torch.as_tensor(self.w2v[str(id_name[0])], dtype=torch.float32)
                
            unique_nodes.append(id_name.unsqueeze(0))

        
        unique_nodes = torch.cat(unique_nodes, dim=0)
        unique_rel = torch.cat(unique_rel, dim=0)
        rels = torch.cat(rels, dim=0)
        labels = [i+1 for i in range(len(real_objs))]
        
        target = {}
        
        target['image_id'] = torch.tensor(int(index), dtype=torch.int64)
        
        target['subject_embedding'] = unique_nodes
        target['object_embedding'] = unique_nodes
        target['relation_embedding'] = rels
        target['relation_unique'] = unique_rel.permute(1,0)
        target['unique_nodes'] = unique_nodes.permute(1,0)
        
        target['boxes'] = boxes
        target['labels'] = torch.tensor(labels, dtype=torch.int64)
        target['edges'] = torch.tensor(triples, dtype=torch.int64)
        

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        
        return len(self.image_ids)

def get_model(num_classes):
    model = lib.fasterrcnn_resnet50_fpn(pretrained=False)

    in_features = model.roi_heads.box_predictor_v2.cls_score.in_features
    num_classes = 2

    model.roi_heads.box_predictor_v2 =FastRCNNPredictorPairedSortedGNNFull(in_features, num_classes)
    
    return model


dataset = COCODataset(glove_vectors, get_transform(train=False))

data_loader_test = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=6,
    collate_fn=utils.collate_fn)

print (len(dataset))


num_classes = 2
print (num_classes)
# # get the model using our helper function
model = get_model(num_classes)


model.to(device)

MODEL_DIR = "saved_models"
num_epochs = 10

model.load_state_dict(torch.load("saved_models/model.pth", map_location=device))


evaluateGNN(model.eval(), data_loader_test, device)
