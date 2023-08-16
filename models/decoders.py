import torch
from torch import nn
import torch.nn.functional as F
import dgl
from torch.nn.utils.rnn import pad_sequence

'''
obsoleted
'''

class InnerProductDecoder(nn.Module):
    def __init__(self, Wq=nn.Identity(), Wk=nn.Identity(), return_feat=False, projector=None):
        super().__init__()
        self.Wq = Wq
        self.Wk = Wk
        self.return_feat = return_feat
        self.projector = projector if return_feat else None

    def forward(self, batched_graph, batched_h):
        # the adjaceny matrix should be computed in each graph, rather than batched graph
        # unbatch the features of nodes from graph
        batch_num_nodes = batched_graph.batch_num_nodes().tolist()
        hidden_list = torch.split(batched_h, batch_num_nodes, dim=0)
        # faster version use torch.nn.utils.rnn.unpad_sequence
        padded_hidden = pad_sequence(hidden_list, batch_first=True)
        q = self.Wq(padded_hidden)
        k = self.Wk(padded_hidden)
        inst_out = torch.bmm(q, k.transpose(1, 2))
        # feature after projector
        feat_out = []
        if self.return_feat:
            feat_out = self.projector(padded_hidden)
        
        return inst_out, feat_out


class InstanceDecoder(nn.Module):
    '''
    obsoleted
    '''
    def __init__(self, Wq=nn.Identity(), Wk=nn.Identity(), return_feat=False, projector=None):
        super().__init__()
        self.Wq = Wq
        self.Wk = Wk
        self.return_feat = return_feat
        self.projector = projector if return_feat else None

    def forward(self, batched_graph, batched_h):
        # the adjaceny matrix should be computed in each graph, rather than batched graph
        batched_graph.ndata['h'] = batched_h
        graph_list = dgl.unbatch(batched_graph)
        # the instance prediction should be computed in each graph
        inst_out = []
        feat_out = []
        for idx, graph in enumerate(graph_list):
            h = graph.ndata['h']
            q = self.Wq(h)
            k = self.Wk(h)
            n = h.shape[0]
            fm1 = torch.unsqueeze(q, dim=2)
            fm2 = torch.unsqueeze(k.T, dim=0)
            fm1 = torch.tile(fm1, dims=[1, 1, n])
            fm2 = torch.tile(fm2, dims=[n, 1, 1])
            sm = torch.square(fm1 - fm2)
            sm = torch.sum(sm, axis=1)
            inst_out.append(sm)
            if self.return_feat:
                p = self.projector(h)
                feat_out.append(p)

        return inst_out, feat_out