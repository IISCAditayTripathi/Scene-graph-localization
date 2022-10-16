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

from PIL import Image, ImageDraw
from lib.faster_rcnn import  FastRCNNPredictorPairedSortedGNNFull
from torchvision.transforms import functional as F
from engine import evaluateGNN
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
                    for obj in rel['objects']:
                        edges += 1
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
        node_names = []
        name = ''
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
                name = name + item['subject_name']+'_'
                subj_name = subject_name
                if len(subject_name)>1:
                    subject_name = torch.as_tensor([self.w2v[str(r)] for r in subject_name], dtype=torch.float32).mean(0)
                else:
                    subject_name = torch.as_tensor(self.w2v[str(subject_name[0])], dtype=torch.float32)

                object_name = item['object_name'].encode().decode("utf-8","ignore").split(' ')
                name = name + item['object_name']+'_'
                obj_name = object_name
                if len(object_name)>1:
                    object_name = torch.as_tensor([self.w2v[str(r)] for r in object_name], dtype=torch.float32).mean(0)
                else:
                    object_name = torch.as_tensor(self.w2v[str(object_name[0])], dtype=torch.float32)

                rel_name = item['predicate'].encode().decode("utf-8","ignore").split(' ')
                
                name = name+item['predicate']
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
                    node_names.append(subj_name)
                    
                    unique_nodes.append(subject_name.unsqueeze(-1))

                if not obj_key in key_happened:
                    key_happened.append(obj_key)
                    boxes.append([obj_box['x'],  obj_box['y'], obj_box['x']+obj_box['w'], obj_box['y']+obj_box['h']])
                    i += 1
                    labels.append(i)
                    node_names.append(obj_name)
                    
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
        target['node_names'] = node_names
        target['name'] = name

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_paths)


def get_model(num_classes):
    model = lib.fasterrcnn_resnet50_fpn(pretrained=False)

    in_features = model.roi_heads.box_predictor_v2.cls_score.in_features
    num_classes = 2

    model.roi_heads.box_predictor_v2 =FastRCNNPredictorPairedSortedGNNFull(in_features, num_classes)
    
    return model


model = 'VGFO'

if model == 'VGFO':
    OBJECTS_FILE = '/path/to/vg/file/localization_vg150vr40_po_test.json'
else:
    OBJECTS_FILE = "/path/to/vg/file/localization_vg150vr40_po_test.json"



dataset = VisualGenomeBansalTrainFullGNNv2(OBJECTS_FILE,glove_vectors,4000, get_transform(train=False))



data_loader_test = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

print (len(dataset))


num_classes = 2
print (num_classes)
# # get the model using our helper function
model = get_model(num_classes)

# torch.hub.load(model, force_reload=True)
if torch.cuda.device_count() > 1:
    print ('Going train with Data Parallel...')
    # model = torch.nn.DataParallel(model)
# move model to the right device

model.to(device)

# In[10]:
MODEL_DIR = "saved_models"
num_epochs = 50

model.load_state_dict(torch.load("saved_models/gnn_model_step_1_full_non_margin_vg_v3_02_max_2.pth", map_location=device))


evaluateGNN(model.eval(), data_loader_test, device)


