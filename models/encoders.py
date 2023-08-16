import torch
from torch import nn
import torch.nn.functional as F
import dgl

from .layers import NodeConv, EdgeConv
from .layers import MLP
from .layers import NodeMPNN, EdgeMPNN, NodeMPNNV2
from .layers import GENConv



class UVNetGraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        input_edge_dim,
        output_dim,
        hidden_dim=64,
        learn_eps=True,
        num_layers=3,
        num_mlp_layers=2,
    ):
        """
        This is the graph neural network used for message-passing features in the
        face-adjacency graph.  (see Section 3.2, Message passing in paper)

        Args:
            input_dim ([type]): [description]
            input_edge_dim ([type]): [description]
            output_dim ([type]): [description]
            hidden_dim (int, optional): [description]. Defaults to 64.
            learn_eps (bool, optional): [description]. Defaults to True.
            num_layers (int, optional): [description]. Defaults to 3.
            num_mlp_layers (int, optional): [description]. Defaults to 2.
        """
        super(UVNetGraphEncoder, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of layers for node and edge feature message passing
        self.node_conv_layers = torch.nn.ModuleList()
        self.edge_conv_layers = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            node_feats = input_dim if layer == 0 else hidden_dim
            edge_feats = input_edge_dim if layer == 0 else hidden_dim
            self.node_conv_layers.append(
                NodeConv(
                    node_feats=node_feats,
                    out_feats=hidden_dim,
                    edge_feats=edge_feats,
                    num_mlp_layers=num_mlp_layers,
                    hidden_mlp_dim=hidden_dim,
                ),
            )
            self.edge_conv_layers.append(
                EdgeConv(
                    edge_feats=edge_feats,
                    out_feats=hidden_dim,
                    node_feats=node_feats,
                    num_mlp_layers=num_mlp_layers,
                    hidden_mlp_dim=hidden_dim,
                )
            )

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

        self.drop1 = nn.Dropout(0.3)
        self.drop = nn.Dropout(0.5)
        self.pool = dgl.nn.MaxPooling()

    def forward(self, g, h, efeat):
        hidden_rep = [h]
        he = efeat

        for i in range(self.num_layers - 1):
            # Update node features
            h = self.node_conv_layers[i](g, h, he)
            # Update edge features
            he = self.edge_conv_layers[i](g, h, he)
            hidden_rep.append(h)

        out = hidden_rep[-1]
        out = self.drop1(out)
        score_over_layer = 0

        # Perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return out, score_over_layer


class GCN(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        num_layers,
        delta,
        mlp_ratio=4,
        drop=0.,
        drop_path=0.,
        conv_on_edge=True,
    ):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        gcn_type = 'GraphConv'
        assert gcn_type in ['GraphConv', 'EdgeConv', 'TAGConv']
        GCNLayer = getattr(dgl.nn, gcn_type)
        # List of layers for node feature message passing
        self.node_conv_layers = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            self.node_conv_layers.append(
                GCNLayer(node_dim, node_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        # linear functions for graph average poolings of output
        self.pool = dgl.nn.AvgPooling()
        self.linear = MLP(1, node_dim, 0, node_dim)

    def forward(self, g, h, he):
        # g = dgl.add_self_loop(g)

        for i in range(self.num_layers):
            # Update node features
            h = self.node_conv_layers[i](g, h)
            h = F.relu(h)
        
        local_feat = h
        # perform graph sum pooling over all nodes
        global_feat = self.linear(self.pool(g, local_feat))
        return local_feat, global_feat


class SAGE(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        num_layers,
        delta,
        mlp_ratio=4,
        drop=0.,
        drop_path=0.,
        conv_on_edge=True,
    ):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        aggregator_type='pool'
        # List of layers for node feature message passing
        self.node_conv_layers = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            self.node_conv_layers.append(
                dgl.nn.SAGEConv(node_dim, node_dim, aggregator_type)
            )

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        # linear functions for graph average poolings of output
        self.pool = dgl.nn.AvgPooling()
        self.linear = MLP(1, node_dim, 0, node_dim)

    def forward(self, g, h, he):
        # g = dgl.add_self_loop(g)

        for i in range(self.num_layers):
            # Update node features
            h = self.node_conv_layers[i](g, h)
            h = F.relu(h)
        
        local_feat = h
        # perform graph sum pooling over all nodes
        global_feat = self.linear(self.pool(g, local_feat))
        return local_feat, global_feat


class GIN(nn.Module):
    def __init__(self, 
                 node_dim,
                 edge_dim,
                 num_layers,
                 delta,
                 mlp_ratio=4,
                 drop=0.,
                 drop_path=0.,
                 conv_on_edge=True):
        super().__init__()
        input_dim = node_dim
        output_dim = node_dim 
        hidden_dim = node_dim
        num_mlp_layers=2
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        # five-layer GCN with l-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1): # excluding the input layer
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(dgl.nn.GINConv(mlp, learn_eps=False, aggregator_type='max')) # set to True if learning epsilon
            self.batch_norms.append(nn.LayerNorm(hidden_dim))
        
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        
        self.drop1 = nn.Dropout(0.3)
        self.drop = nn.Dropout(0.5)
        self.pool = dgl.nn.AvgPooling() # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, h, he):
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        
        out = hidden_rep[-1]
        out = self.drop1(out)
        score_over_layer = 0

        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))

        return out, score_over_layer

    
class GAT(nn.Module):
    def __init__(self, 
                 node_dim,
                 edge_dim,
                 num_layers,
                 delta,
                 mlp_ratio=4,
                 drop=0.,
                 drop_path=0.,
                 conv_on_edge=True):
        super().__init__()
        in_size = node_dim
        hid_size = node_dim
        out_size = node_dim
        heads=[4, 4, 6]
        self.gat_layers = nn.ModuleList()
        # three-layer GAT
        self.gat_layers.append(
            dgl.nn.GATConv(in_size, hid_size, heads[0], activation=F.elu)
        )
        self.gat_layers.append(
            dgl.nn.GATConv(
                hid_size * heads[0],
                hid_size,
                heads[1],
                residual=True,
                activation=F.elu,
            )
        )
        self.gat_layers.append(
            dgl.nn.GATConv(
                hid_size * heads[1],
                out_size,
                heads[2],
                residual=True,
                activation=None,
            )
        )
        
        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        # linear functions for graph average poolings of output
        self.pool = dgl.nn.AvgPooling()
        self.linear = MLP(1, node_dim, 0, node_dim)

    def forward(self, g, h, he):
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 2:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        local_feat = h
        # perform graph sum pooling over all nodes
        global_feat = self.linear(self.pool(g, local_feat))
        return local_feat, global_feat

    
class GATv2(nn.Module):
    def __init__(self,
                 node_dim,
                 edge_dim,
                 num_layers,
                 delta,
                 mlp_ratio=4,
                 drop=0.,
                 drop_path=0.,
                 conv_on_edge=True):
        super(GATv2, self).__init__()
        num_layers = 1
        in_dim = node_dim
        num_hidden = node_dim
        num_classes = node_dim
        num_heads = 8
        num_out_heads= 1
        heads = ([num_heads] * num_layers) + [num_out_heads]
        activation = F.elu
        feat_drop = 0.25
        attn_drop = 0.25
        negative_slope = 0.2
        residual = True
        
        self.num_layers = num_layers
        self.gatv2_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gatv2_layers.append(
            dgl.nn.GATv2Conv(
                in_dim,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
                bias=False,
                share_weights=True,
            )
        )
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gatv2_layers.append(
                dgl.nn.GATv2Conv(
                    num_hidden * heads[l - 1],
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    bias=False,
                    share_weights=True,
                )
            )
        # output projection
        self.gatv2_layers.append(
            dgl.nn.GATv2Conv(
                num_hidden * heads[-2],
                num_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
                bias=False,
                share_weights=True,
            )
        )
        
        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        # linear functions for graph average poolings of output
        self.pool = dgl.nn.AvgPooling()
        self.linear = MLP(1, node_dim, 0, node_dim)

    def forward(self, g, h, he):
        for l in range(self.num_layers):
            h = self.gatv2_layers[l](g, h).flatten(1)
        # output projection
        local_feat = self.gatv2_layers[-1](g, h).mean(1)

        # perform graph sum pooling over all nodes
        global_feat = self.linear(self.pool(g, local_feat))
        return local_feat, global_feat

    
class DeeperGCN(nn.Module):
    r"""
    Description
    -----------
    Introduced in "DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>"
    Parameters
    ----------
    node_feat_dim: int
        Size of node feature.
    edge_feat_dim: int
        Size of edge feature.
    hid_dim: int
        Size of hidden representations.
    out_dim: int
        Size of output.
    num_layers: int
        Number of graph convolutional layers.
    dropout: float
        Dropout rate. Default is 0.
    beta: float
        A continuous variable called an inverse temperature. Default is 1.0.
    learn_beta: bool
        Whether beta is a learnable weight. Default is False.
    aggr: str
        Type of aggregation. Default is 'softmax'.
    mlp_layers: int
        Number of MLP layers in message normalization. Default is 1.
    """

    def __init__(self,
                 node_dim,
                 edge_dim,
                 num_layers,
                 delta,
                 mlp_ratio=4,
                 drop=0.,
                 drop_path=0.,
                 conv_on_edge=True):
        super(DeeperGCN, self).__init__()
        node_feat_dim=node_dim
        edge_feat_dim=edge_dim
        hid_dim=node_dim
        out_dim=node_dim
        dropout=0.2
        beta=1.0
        learn_beta=False
        aggr="softmax"
        mlp_layers=1
        self.num_layers = num_layers
        self.dropout = dropout
        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            conv = GENConv(
                in_dim=hid_dim,
                out_dim=hid_dim,
                aggregator=aggr,
                beta=beta,
                learn_beta=learn_beta,
                mlp_layers=mlp_layers,
            )

            self.gcns.append(conv)
            self.norms.append(nn.BatchNorm1d(hid_dim, affine=True))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        # linear functions for graph average poolings of output
        self.pool = dgl.nn.AvgPooling()
        self.linear = MLP(1, node_dim, 0, node_dim)

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            hv = node_feats
            he = edge_feats

            for layer in range(self.num_layers):
                hv1 = self.norms[layer](hv)
                hv1 = F.relu(hv1)
                hv1 = F.dropout(hv1, p=self.dropout, training=self.training)
                hv = self.gcns[layer](g, hv1, he) + hv

            local_feat = hv
            # perform graph sum pooling over all nodes
            global_feat = self.linear(self.pool(g, local_feat))
            return local_feat, global_feat

        
class AAGNetGraphEncoder(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        num_layers,
        delta,
        mlp_ratio=4,
        drop=0.,
        drop_path=0.,
        conv_on_edge=True
    ):
        """

        Args:
            input_dim (int): [description]
            input_edge_dim (int): [description]
            output_dim (int): [description]
            num_layers (int, optional): [description].
        """
        super(AAGNetGraphEncoder, self).__init__()
        self.num_layers = num_layers
        self.conv_on_edge = conv_on_edge
        self.node_convs = nn.ModuleList()
        self.edge_convs = nn.ModuleList()        
        # since 2nd layer, the subsequent layers are share-weight
        for _ in range(2):
            if self.conv_on_edge:
                self.edge_convs.append(
                    EdgeMPNN(node_dim, edge_dim, mlp_ratio, drop, drop_path))
            self.node_convs.append(
                NodeMPNN(node_dim, edge_dim, delta, mlp_ratio, drop, drop_path))

        self.post_norm = nn.LayerNorm(node_dim)
        # linear functions for graph average poolings of output
        self.pool = dgl.nn.AvgPooling()
        self.linear = MLP(1, node_dim, 0, node_dim, nn.LayerNorm, True)
    
    def forward(self, g, h, he):
        # first layer
        if self.conv_on_edge:
            he = self.edge_convs[0](g, h, he)
        h = self.node_convs[0](g, h, he)
        
        # subsequent share-weight layer
        for i in range(self.num_layers-1):
            if self.conv_on_edge:
                he = self.edge_convs[1](g, h, he)
            h = self.node_convs[1](g, h, he)
        
        local_feat = self.post_norm(h)
        # perform graph sum pooling over all nodes
        global_feat = self.linear(self.pool(g, local_feat))
        return local_feat, global_feat

    
class AAGNetGraphEncoderV2(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        num_layers,
        delta,
        mlp_ratio=4,
        drop=0.,
        drop_path=0.,
        conv_on_edge=True
    ):
        """

        Args:
            input_dim (int): [description]
            input_edge_dim (int): [description]
            output_dim (int): [description]
            num_layers (int, optional): [description].
        """
        super(AAGNetGraphEncoderV2, self).__init__()
        self.num_layers = num_layers
        self.node_convs = nn.ModuleList()
        self.edge_convs = nn.ModuleList()        
        # since 2nd layer, the subsequent layers are share-weight
        for _ in range(2):
            self.node_convs.append(
                NodeMPNNV2(node_dim, edge_dim, delta, mlp_ratio, drop, drop_path))

        self.post_norm = nn.LayerNorm(node_dim)
        # linear functions for graph average poolings of output
        self.pool = dgl.nn.AvgPooling()
        self.linear = MLP(1, node_dim, 0, node_dim, nn.LayerNorm, True)
    
    def forward(self, g, h, he):
        # first layer
        h = self.node_convs[0](g, h, he)
        
        # subsequent share-weight layer
        for i in range(self.num_layers-1):
            h = self.node_convs[1](g, h, he)
        
        local_feat = self.post_norm(h)
        # perform graph sum pooling over all nodes
        global_feat = self.linear(self.pool(g, local_feat))
        return local_feat, global_feat
