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
from collections import defaultdict

import utils
import torch
import torchvision
import torch.utils.data as data

from PIL import Image, ImageDraw
from lib.faster_rcnn import  FastRCNNPredictorPairedSortedGNNFull
from torchvision.transforms import functional as F
from engine import train_one_epoch
import lib
import gensim
from gensim import downloader
import torch.multiprocessing
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
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)



class VisualGenomeBansalTrainFullGNNv2(data.Dataset):
    def __init__(self, object_file_path,glove_vectors,length, transforms=None):

        self.image_dir = "/path/to/vg/images/VG_100K"
        
        self.data = {}
        object_data = json.load(open(object_file_path))

        self.image_paths = []
        self.image_items = {}

        self.image_objects = object_data['objects']
        self.image_data = object_data['image_data']
        self.transforms = transforms

        for data_item in self.image_data:
            img_id = data_item['image_id']
            imdb_id = data_item['imdb_id']
            
            for rels in data_item['connected_scene_graphs']:
                graph = defaultdict(list)
                edges = 0
                for rel in rels:
                    edges += 1
                    for obj in rel['objects']:
                        graph[obj['subject_id']].append({'object_id':obj['object_id'], 'subject_name':rel['subject_name'], 'object_name': rel['object_name'], 'predicate':rel['predicate']})
                
                self.image_paths.append([img_id, graph])
        
        print(len(self.image_paths))


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
        s = self.image_paths[index]

        img = self.load_image(self.image_paths[index][0])

        data_object = s[1]

        key2id = {}
        # id2key = {}
        boxes = []
        labels = []
        subjects = []
        objects = []
        predicates = []
        i = 0
        edges = []
        unique_rel = []
        unique_rel_name = []
        key_happened = []
        unique_nodes = []
        key2id_pred = {}
        for key, items in data_object.items():
            
            for item in items:
                subj_id = key
                obj_id = item['object_id']
                if subj_id in key2id.keys():
                    subj_key = key2id[subj_id]
                else: 
                    key2id[subj_id] = len(key2id)
                    subj_key = key2id[subj_id]
                if obj_id in key2id.keys():
                    obj_key = key2id[obj_id]
                else:
                    key2id[obj_id] = len(key2id)
                    obj_key = key2id[obj_id]
                pred_key = item['predicate']
                if pred_key in key2id_pred.keys():
                    pred_key = key2id_pred[pred_key]
                else:
                    key2id_pred[pred_key] = len(key2id_pred)
                    pred_key = key2id_pred[pred_key]
                
                edges.append([subj_key, pred_key, obj_key])
                subj_box = self.image_objects[str(subj_id)]['bbox_orig']
                obj_box = self.image_objects[str(obj_id)]['bbox_orig']

                
                subject_name = item['subject_name'].encode().decode("utf-8","ignore").split(' ')
                if len(subject_name)>1:
                    subject_name = torch.as_tensor([self.w2v[str(r)] for r in subject_name], dtype=torch.float32).mean(0)
                else:
                    subject_name = torch.as_tensor(self.w2v[str(subject_name[0])], dtype=torch.float32)

                object_name = item['object_name'].encode().decode("utf-8","ignore").split(' ')
                
                if len(object_name)>1:
                    object_name = torch.as_tensor([self.w2v[str(r)] for r in object_name], dtype=torch.float32).mean(0)
                else:
                    object_name = torch.as_tensor(self.w2v[str(object_name[0])], dtype=torch.float32)

                rel_name = item['predicate'].encode().decode("utf-8","ignore").split(' ')
                
                if len(rel_name) >1:
                    rel_emb = torch.as_tensor([self.w2v[str(r)] for r in rel_name], dtype=torch.float32).mean(0)

                else:
                    rel_emb = torch.as_tensor(self.w2v[str(rel_name[0])], dtype=torch.float32)
                if item['predicate'] in unique_rel_name:
                    pass
                else:
                    unique_rel.append(rel_emb.unsqueeze(-1))
                    unique_rel_name.append(item['predicate'])
                
                if not subj_key in key_happened:
                    key_happened.append(subj_key)
                    boxes.append([subj_box['x'],  subj_box['y'], subj_box['x']+subj_box['w'], subj_box['y']+subj_box['h']])
                    i = i+1
                    labels.append(i)
                    
                    unique_nodes.append(subject_name.unsqueeze(-1))

                if not obj_key in key_happened:
                    key_happened.append(obj_key)
                    boxes.append([obj_box['x'],  obj_box['y'], obj_box['x']+obj_box['w'], obj_box['y']+obj_box['h']])
                    i += 1
                    labels.append(i)
                    
                    unique_nodes.append(object_name.unsqueeze(-1))


                subjects.append(subject_name.unsqueeze(-1))
                objects.append(object_name.unsqueeze(-1))
                predicates.append(rel_emb.unsqueeze(-1))
        
        target = {}
        
        target['image_id'] = torch.tensor(int(index), dtype=torch.int64)
        
        target['subject_embedding'] = torch.cat(subjects,dim=-1)
        target['object_embedding'] = torch.cat(objects, dim=-1)
        target['relation_embedding'] = torch.cat(predicates, dim=-1)
        target['relation_unique'] = torch.cat(unique_rel, dim=-1)
        target['unique_nodes'] = torch.cat(unique_nodes, dim=-1)

        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.tensor(labels, dtype=torch.int64)
        target['edges'] = torch.tensor(edges, dtype=torch.int64)


        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_paths)

def get_model(num_classes):
    model = lib.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor_v2.cls_score.in_features
    num_classes = 2

    model.roi_heads.box_predictor_v2 =FastRCNNPredictorPairedSortedGNNFull(in_features, num_classes)
    
    return model


# Parameters
model = 'VGFO'
lr = 0.02
momentum = 0.9
weight_decay = 0.0005
lr_backbone = 0.00001
batch_size = 7
num_workers = 8
shuffle = True  # Shuffle the data
step = 2
gamma = 0.1
num_epochs = 10

if model == 'VGFO':
    OBJECTS_FILE = '/path/to/vg/file/localization_vg150vr40_po_train.json'
else:
    OBJECTS_FILE = "/path/to/vg/file/localization_vg150vr40_po_train.json"



dataset = VisualGenomeBansalTrainFullGNNv2(OBJECTS_FILE,glove_vectors,4000, get_transform(train=True))


dataset_train = dataset
dataset.transforms = get_transform(train=True)

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
    collate_fn=utils.collate_fn)


num_classes = 2
print (num_classes)
# # get the model using our helper function
model = get_model(num_classes)


param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n  and p.requires_grad
            ]
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": lr_backbone,
        }
    ]
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("number of params:", n_parameters)
# torch.hub.load(model, force_reload=True)
if torch.cuda.device_count() > 1:
    print ('Going train with Data Parallel...')
    model = torch.nn.DataParallel(model)
# move model to the right device
model.to(device)

# construct an optimizer

optimizer = torch.optim.SGD(param_dicts, lr=lr,                  #lr=0.0005
                            momentum=momentum, weight_decay=weight_decay)


# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=step,    #stepsize=5
                                               gamma=gamma)


# In[10]:
MODEL_DIR = "saved_models"


for epoch in range(num_epochs):
    
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model_epoch_%s.pth"%(epoch)))
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
    print("saving model_epoch_"+str(epoch))
    #evaluate model
    # update the learning rate
    lr_scheduler.step()


