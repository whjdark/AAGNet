import pathlib
import json
import os
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import dgl
import dgl.function as fn 

from ..dataloader.mfinstseg import MFInstSegDataset


if __name__ == '__main__':
    dataset = MFInstSegDataset(root_dir='E:\\AAGNet\\dataset\\data', split='all', center_and_scale=False)
    n_classes = dataset.num_classes()
    d_list = []
    for data in dataset:
        graphs = data["graph"].to('cuda')
        d = torch.sparse.sum(graphs.adjacency_matrix(transpose=True), dim=-1).to_dense()
        d_list.append(d)
    d_list = torch.cat(d_list)
    delta = torch.mean(torch.log(d_list + 1))
    print('log(d+1):', delta)