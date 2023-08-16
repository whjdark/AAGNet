import torch
from torch import nn
import torch.nn.functional as F
import dgl


'''
obsoleted
'''

class InstanceLoss(nn.Module):
    def __init__(self, inst_loss=nn.BCEWithLogitsLoss, reduction='mean', device='cpu') -> None:
        super(InstanceLoss, self).__init__()
        self.inst_loss = inst_loss(reduction=reduction)
        self.device = device

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> tuple:
        # split the given batched instance label graph into a list of small ones
        instance_label_list = dgl.unbatch(targets)
        assert len(inputs) == len(instance_label_list)
        num_graph_per_batch = len(inputs)
        # the instance losses should be computed in each graph
        instance_losses = torch.zeros((num_graph_per_batch), dtype=torch.float32, device=self.device)
        for idx, data in enumerate(zip(inputs, instance_label_list)):
            pred, label = data
            # get the relation matirx (adjacency matrix) from instance label graph
            label_adj = label.adj().to_dense().to(self.device)
            instance_losses[idx] = self.inst_loss(pred, label_adj.float())
        return torch.mean(instance_losses)