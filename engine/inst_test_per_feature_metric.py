import pathlib
import functools
import json
import random
import os
import time
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import dgl
import dgl.function as fn 


from dataloader.mfinstseg import MFInstSegDataset
from models.inst_segmentors import AAGNetSegmentor
from utils.misc import seed_torch


EPS = 1e-6
INST_THRES = 0.5
BOTTOM_THRES = 0.5

feat_names = ['chamfer', 'through_hole', 'triangular_passage', 'rectangular_passage', '6sides_passage',
              'triangular_through_slot', 'rectangular_through_slot', 'circular_through_slot',
              'rectangular_through_step', '2sides_through_step', 'slanted_through_step', 'Oring', 'blind_hole',
              'triangular_pocket', 'rectangular_pocket', '6sides_pocket', 'circular_end_pocket',
              'rectangular_blind_slot', 'v_circular_end_blind_slot', 'h_circular_end_blind_slot',
              'triangular_blind_step', 'circular_blind_step', 'rectangular_blind_step', 'round', 'stock'
             ]


def print_class_metric(metric):
    string = ''
    for i in range(len(metric)):
        string += feat_names[i] + ': ' + str(metric[i]) + ', '
    print(string)


class FeatureInstance():
    def __init__(self, name:int = None, 
                       faces:np.array = None, 
                       bottoms:list = None):
        self.name = name
        self.faces = faces
        self.bottoms = bottoms


def parser_label(inst_label, seg_label, bottom_label):
    label_list = []
    # parse instance label
    inst_label = np.array(inst_label, dtype=np.uint8)[0]
    used_faces = []
    for row_idx, row in enumerate(inst_label):
        if np.sum(row) == 0:
            # stock face, no linked face, so the sum of the column is 0
            continue
        # when I_ij = 1 mean face_i is linked with face_j
        # so can get the indices of linked faces in a instance
        linked_face_idxs = np.where(row==1)[0]
        # used
        if len(set(linked_face_idxs).intersection(set(used_faces))) > 0:
            # the face has been assigned to a instance
            continue
        # create new feature
        new_feat = FeatureInstance()
        new_feat.faces = linked_face_idxs
        used_faces.extend(linked_face_idxs)
        # all the linked faces in a instance 
        # have the same segmentation label
        # so get the name of the instance
        a_face_id = new_feat.faces[0]
        seg_id = seg_label[int(a_face_id)]
        new_feat.name = seg_id
        # get the bottom face segmentation label
        # new_feat.bottoms = np.where(bottom_label==1)[0]
        # # add new feature into list and used face counter
        label_list.append(new_feat)
    
    return label_list


def post_process(out, inst_thres, bottom_thres):
    seg_out, inst_out, bottom_out = out
    # post-processing for semantic segmentation 
    # face_logits = torch.argmax(seg_out, dim=1)
    face_logits = seg_out.cpu().numpy()

    # post-processing for instance segmentation 
    inst_out = inst_out[0] # inst_out is a list
    inst_out = inst_out.sigmoid()
    adj = inst_out > inst_thres
    adj = adj.cpu().numpy().astype('int32')

    # post-processing for bottom classification 
    # bottom_out = bottom_out.sigmoid()
    # bottom_logits = bottom_out > bottom_thres
    # bottom_logits = bottom_logits.cpu().numpy()
    
    # Identify individual proposals of each feature
    proposals = set() # use to delete repeat proposals
    # record whether the face belongs to a instance
    used_flags = np.zeros(adj.shape[0], dtype=np.bool_)
    for row_idx, row in enumerate(adj):
        if used_flags[row_idx]:
            # the face has been assigned to a instance
            continue
        if np.sum(row) <= EPS: 
            # stock face, no linked face, so the sum of the column is 0
            continue
        # non-stock face
        proposal = set() # use to delete repeat faces
        for col_idx, item in enumerate(row):
            if used_flags[col_idx]:
                # the face has been assigned to a proposal
                continue
            if item: # have connections with currect face
                proposal.add(col_idx)
                used_flags[col_idx] = True
        if len(proposal) > 0:
            proposals.add(frozenset(proposal)) # frozenset is a hashable set
    # TODO: better post-process
    
    # save new results
    features_list = []
    for instance in proposals:
        instance = list(instance)
        # sum voting for the class of the instance
        sum_inst_logit = 0
        for face in instance:
            sum_inst_logit += face_logits[face]
        # the index of max score is the class of the instance
        inst_logit = np.argmax(sum_inst_logit)
        if inst_logit == 24:
            # is stock, ignore
            continue
        # get instance label name from face_logits

        inst_name = inst_logit
        # get the bottom faces
        # bottom_faces = []
        # for face_idx in instance:
        #     if bottom_logits[face_idx]:
        #         bottom_faces.append(face_idx)
        features_list.append(
            FeatureInstance(name=inst_name, faces=np.array(instance)))
    
    return features_list


def cal_recognition_performance(feature_list, label_list):
    # one hot encoding
    pred = np.zeros(24, dtype=int)
    gt = np.zeros(24, dtype=int)
    for feature in feature_list:
        pred[feature.name] += 1
    for label in label_list:
        gt[label.name] += 1
    tp = np.minimum(gt, pred)

    return pred, gt, tp


def cal_localization_performance(feature_list, label_list):
    # one hot encoding
    pred = np.zeros(24, dtype=int)
    gt = np.zeros(24, dtype=int)
    for feature in feature_list:
        pred[feature.name] += 1
    for label in label_list:
        gt[label.name] += 1
    
    # sort the feature_list and label_list by name
    feature_list.sort(key=lambda x: x.name)
    label_list.sort(key=lambda x: x.name)
    tp = np.zeros(24, dtype=int)

    
    found_lbl = np.zeros(len(label_list))
    # for each detection
    for pred_i in range(len(feature_list)):
        pred_name = feature_list[pred_i].name

        #among the ground-truths, choose one that belongs to the same class and has the highest IoU with the detection        
        for lbl_i in range(len(label_list)):  
            lbl_name = label_list[lbl_i].name
        
            if pred_name != lbl_name or found_lbl[lbl_i] == 1:
                    continue
            
            # compute IoU
            pred_faces = feature_list[pred_i].faces
            lbl_faces = label_list[lbl_i].faces
            intersection = np.intersect1d(pred_faces, lbl_faces)
            union = np.union1d(pred_faces, lbl_faces)
            iou = len(intersection) / len(union)

            # when IOU == 1, the detection is correct
            # else the detection is wrong
            if iou >= 1.0 - EPS:
                found_lbl[lbl_i] = 1
                tp[pred_name] += 1
                break
    
    # when tp gt not equal, print the detail
    # if not np.all(tp == gt):
    #     for feature in feature_list:
    #         feature.faces.sort()
    #         print('feature', feature.name, feature.faces)
    #     for label in label_list:
    #         label.faces.sort()
    #         print('label', label.name, label.faces)

    #     print('tp', tp)
    #     print('pd', pred)
    #     print('gt', gt)

    return pred, gt, tp


def eval_metric(pre, trul, tp):
    precision = tp / pre
    recall = tp / trul
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    # if the gt[i] == 0, mean class i is not in the ground truth
    # so the precision and recall of class i is not defined
    # so set the precision and recall of class i to 1
    precision[trul == 0] = 1
    recall[trul == 0] = 1

    return precision, recall


if __name__ == '__main__':
    # track hyperparameters and run metadata
    torch.set_float32_matmul_precision("high") # may be faster if GPU support TF32
    config={
        "edge_attr_dim": 12,
        "node_attr_dim": 10,
        "edge_attr_emb": 64, # recommend: 64
        "node_attr_emb": 64, # recommend: 64
        "edge_grid_dim": 0, 
        "node_grid_dim": 7,
        "edge_grid_emb": 0, 
        "node_grid_emb": 64, # recommend: 64
        "num_layers": 3, # recommend: 3
        "delta": 2, # obsolete
        "mlp_ratio": 2,
        "drop": 0.25, 
        "drop_path": 0.25,
        "head_hidden_dim": 64,
        "conv_on_edge": False,
        "use_uv_gird": True,
        "use_edge_attr": True,
        "use_face_attr": True,

        "seed": 42,
        "device": 'cuda',
        "architecture": "AAGNetGraphEncoder", # recommend: AAGNetGraphEncoder option: GCN SAGE GIN GAT GATv2 DeeperGCN AAGNetGraphEncoder AAGNetGraphEncoderV2
        "dataset_type": "full",
        "dataset": "E:\\traning_data\\data2",

        "epochs": 100,
        "lr": 1e-2,
        "weight_decay": 1e-2,
        "batch_size": 1,
        "ema_decay_per_epoch": 1. / 2.,
        "seg_a": 1.,
        "inst_a": 1.,
        "bottom_a": 1.,
        }

    seed_torch(config['seed'])
    device = config['device']
    dataset = config['dataset']
    dataset_type = config['dataset_type']
    n_classes = MFInstSegDataset.num_classes(dataset_type)

    model = AAGNetSegmentor(num_classes=n_classes,
                            arch=config['architecture'],
                            edge_attr_dim=config['edge_attr_dim'], 
                            node_attr_dim=config['node_attr_dim'], 
                            edge_attr_emb=config['edge_attr_emb'], 
                            node_attr_emb=config['node_attr_emb'],
                            edge_grid_dim=config['edge_grid_dim'], 
                            node_grid_dim=config['node_grid_dim'], 
                            edge_grid_emb=config['edge_grid_emb'], 
                            node_grid_emb=config['node_grid_emb'], 
                            num_layers=config['num_layers'], 
                            delta=config['delta'], 
                            mlp_ratio=config['mlp_ratio'], 
                            drop=config['drop'], 
                            drop_path=config['drop_path'], 
                            head_hidden_dim=config['head_hidden_dim'],
                            conv_on_edge=config['conv_on_edge'],
                            use_uv_gird=config['use_uv_gird'],
                            use_edge_attr=config['use_edge_attr'],
                            use_face_attr=config['use_face_attr'],)
    model = model.to(device)

    model_param = torch.load(".\\weights\\weight_on_MFInstseg.pth", map_location=device)
    model.load_state_dict(model_param) 

    test_dataset = MFInstSegDataset(root_dir=dataset, split='test', 
                                     center_and_scale=False, normalize=True, random_rotate=False,
                                     dataset_type=dataset_type, num_threads=4)
    test_loader = test_dataset.get_dataloader(batch_size=config['batch_size'], pin_memory=True)

    rec_predictions = np.zeros(24, dtype=int)
    rec_truelabels = np.zeros(24, dtype=int)
    rec_truepositives = np.zeros(24, dtype=int)

    loc_predictions = np.zeros(24, dtype=int)
    loc_truelabels = np.zeros(24, dtype=int)
    loc_truepositives = np.zeros(24, dtype=int)

    time_accumulator = 0
    with torch.no_grad():
        print(f'------------- Now start testing ------------- ')
        model.eval()
        # test_per_inst_acc = []
        test_losses = []
        for data in tqdm(test_loader):
            graphs = data["graph"].to(device, non_blocking=True)
            inst_label = data["inst_labels"].cpu().numpy()
            seg_label = graphs.ndata["seg_y"].cpu().numpy()
            bottom_label = graphs.ndata["bottom_y"].cpu().numpy()
            
            # Forward pass
            start_time = time.time()
            out = model(graphs)
            features = post_process(out, inst_thres=INST_THRES, bottom_thres=BOTTOM_THRES)
            time_accumulator += time.time() - start_time

            # calculate recognition performance
            labels = parser_label(inst_label, seg_label, bottom_label)
            pred, gt, tp = cal_recognition_performance(features, labels)
            rec_predictions += pred
            rec_truelabels += gt
            rec_truepositives += tp

            # calculate localization performance
            pred, gt, tp = cal_localization_performance(features, labels)
            loc_predictions += pred
            loc_truelabels += gt
            loc_truepositives += tp

        print('------------- recognition performance------------- ')
        print('rec_pred', rec_predictions)
        print('rec_true', rec_truelabels)
        print('rec_trpo', rec_truepositives)
        precision, recall = eval_metric(rec_predictions, rec_truelabels, rec_truepositives)
        print('recognition Precision scores')
        # print precision for each class
        print_class_metric(precision)
        precision = precision.mean()
        print('AVG recognition Precision:', precision)
        print('recognition Recall scores')
        # print recall for each class
        print_class_metric(recall)
        recall = recall.mean()
        print('AVG recognition Precision:', recall)
        print('recognition F scores')
        rec_F = (2*recall*precision)/(recall+precision)
        print(rec_F)

        print('------------- localization performance------------- ')
        print('loc_pred', loc_predictions)
        print('loc_true', loc_truelabels)
        print('loc_trpo', loc_truepositives)
        precision, recall = eval_metric(loc_predictions, loc_truelabels, loc_truepositives)
        print('localization Precision scores')
        # print precision for each class
        print_class_metric(precision)
        precision = precision.mean()
        print('AVG localization Precision:', precision)
        print('localization Recall scores')
        # print recall for each class
        print_class_metric(recall)
        recall = recall.mean()
        print('AVG localization Precision:', recall)
        print('localization F scores')
        loc_F = (2*recall*precision)/(recall+precision)
        print(loc_F)

        print('------------- average time cost per STEP------------- ')
        print(time_accumulator / len(test_loader))

        print('------------- Final ------------- ')
        print('rec F scores(%):', rec_F*100)
        print('loc F scores(%):', loc_F*100)