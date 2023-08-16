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
import torchmetrics
from torchmetrics.classification import (
    MulticlassAccuracy, 
    BinaryAccuracy, 
    BinaryF1Score, 
    BinaryJaccardIndex, 
    MulticlassJaccardIndex,
    BinaryAveragePrecision)

from dataloader.mfinstseg import MFInstSegDataset
from models.inst_segmentors import AAGNetSegmentor
from utils.misc import seed_torch



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
        "batch_size": 256,
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

    model_param = torch.load("./weights/weight_on_MFInstseg.pth", map_location=device)
    model.load_state_dict(model_param)

    test_dataset = MFInstSegDataset(root_dir=dataset, split='test', 
                                     center_and_scale=False, normalize=True, random_rotate=False,
                                     dataset_type=dataset_type, num_threads=8)
    test_loader = test_dataset.get_dataloader(batch_size=config['batch_size'], pin_memory=True)

    seg_loss = nn.CrossEntropyLoss()
    instance_loss = nn.BCEWithLogitsLoss()
    bottom_loss = nn.BCEWithLogitsLoss()
    
    test_seg_acc = MulticlassAccuracy(num_classes=n_classes).to(device)
    test_inst_acc = BinaryAccuracy().to(device)
    test_bottom_acc = BinaryAccuracy().to(device)
    
    test_seg_iou = MulticlassJaccardIndex(num_classes=n_classes).to(device)
    test_inst_f1 = BinaryF1Score().to(device)
    # test_inst_ap = BinaryAveragePrecision().to(device)
    test_bottom_iou = BinaryJaccardIndex().to(device)

    best_acc = 0.
    with torch.no_grad():
        print(f'------------- Now start testing ------------- ')
        model.eval()
        # test_per_inst_acc = []
        test_losses = []
        for data in tqdm(test_loader):
            graphs = data["graph"].to(device, non_blocking=True)
            inst_label = data["inst_labels"].to(device, non_blocking=True)
            seg_label = graphs.ndata["seg_y"]
            bottom_label = graphs.ndata["bottom_y"]
            
            # Forward pass
            seg_pred, inst_pred, bottom_pred = model(graphs)

            loss_seg = seg_loss(seg_pred, seg_label)
            loss_inst = instance_loss(inst_pred, inst_label)
            loss_bottom = bottom_loss(bottom_pred, bottom_label)
            loss = config['seg_a'] * loss_seg + \
                   config['inst_a'] * loss_inst + \
                   config['bottom_a'] * loss_bottom
            test_losses.append(loss.item())
            test_seg_acc.update(seg_pred, seg_label)
            test_seg_iou.update(seg_pred, seg_label)
            test_inst_acc.update(inst_pred, inst_label)
            test_inst_f1.update(inst_pred, inst_label)
            test_bottom_acc.update(bottom_pred, bottom_label)
            test_bottom_iou.update(bottom_pred, bottom_label)
        
        # batch end
        mean_test_loss = np.mean(test_losses).item()
        mean_test_seg_acc = test_seg_acc.compute().item()
        mean_test_seg_iou = test_seg_iou.compute().item()
        mean_test_inst_acc = test_inst_acc.compute().item()
        mean_test_inst_f1 = test_inst_f1.compute().item()
        mean_test_bottom_acc = test_bottom_acc.compute().item()
        mean_test_bottom_iou = test_bottom_iou.compute().item()
        
        print(f'test_loss : {mean_test_loss}, \
              test_seg_acc: {mean_test_seg_acc}, \
              test_seg_iou: {mean_test_seg_iou}, \
              test_inst_acc: {mean_test_inst_acc}, \
              test_inst_f1: {mean_test_inst_f1}, \
              test_bottom_acc: {mean_test_bottom_acc}, \
              test_bottom_iou: {mean_test_bottom_iou}')