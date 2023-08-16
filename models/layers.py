import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import dgl
import dgl.function as fn

from .pnaconv import PNAConvTower



class NonLinearClassifier(nn.Module):
    def __init__(self, 
                 input_dim, 
                 num_classes, 
                 hidden_dim=512, 
                 dropout=0.3, 
                 act=nn.ReLU):
        """
        A 3-layer MLP with linear outputs

        Args:
            input_dim (int): Dimension of the input tensor 
            num_classes (int): Dimension of the output logits
            dropout (float, optional): Dropout used after each linear layer. Defaults to 0.3.
            act (nn.Module, optional): Activation function. Defaults to nn.ReLU.
        """
        super().__init__()
        second_hidden_dim = hidden_dim // 2
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(hidden_dim, second_hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(second_hidden_dim)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(second_hidden_dim, num_classes)
        self.act = act()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        """
        Forward pass

        Args:
            inp (torch.tensor): Inputs features to be mapped to logits
                                (batch_size x input_dim)

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        x = self.act(self.bn1(self.linear1(inp)))
        x = self.dp1(x)
        x = self.act(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class MLP(nn.Module):
    """"""

    def __init__(self, 
                 num_layers, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 norm=nn.BatchNorm1d,
                 last_norm=False, 
                 act=nn.ReLU):
        """
        MLP with linear output
        Args:
            num_layers (int): The number of linear layers in the MLP
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden feature dimensions for all hidden layers
            output_dim (int): Output feature dimension
            act (nn.Module, optional): Activation function. Defaults to nn.ReLU.
        Raises:
            ValueError: If the given number of layers is <1
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.norm = norm
        self.last_norm = last_norm
        self.act = act()

        if num_layers < 1:
            raise ValueError("Number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
            self.norms.append(self.norm(hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
                self.norms.append(self.norm(hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
        
        if self.last_norm:
            self.post_norm = self.norm(output_dim)
        
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            out = self.linear(x)
            return self.post_norm(out) if self.last_norm else out
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = self.act(self.norms[i](self.linears[i](h)))
            out = self.linears[-1](h)
            return self.post_norm(out) if self.last_norm else out


class EdgeConv(nn.Module):
    def __init__(
        self,
        edge_feats,
        out_feats,
        node_feats,
        num_mlp_layers=2,
        hidden_mlp_dim=64,
    ):
        """
        This module implements Eq. 2 from the paper where the edge features are
        updated using the node features at the endpoints.
        Args:
            edge_feats (int): Input edge feature dimension
            out_feats (int): Output feature deimension
            node_feats (int): Input node feature dimension
            num_mlp_layers (int, optional): Number of layers used in the MLP. Defaults to 2.
            hidden_mlp_dim (int, optional): Hidden feature dimension in the MLP. Defaults to 64.
        """
        super(EdgeConv, self).__init__()
        self.proj = MLP(1, node_feats, hidden_mlp_dim, edge_feats)
        self.mlp = MLP(num_mlp_layers, edge_feats, hidden_mlp_dim, out_feats)
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, graph, nfeat, efeat):
        src, dst = graph.edges()
        proj1, proj2 = self.proj(nfeat[src]), self.proj(nfeat[dst])
        agg = proj1 + proj2
        h = self.mlp((1 + self.eps) * efeat + agg)
        h = F.leaky_relu(self.batchnorm(h))

        return h


class NodeConv(nn.Module):
    def __init__(
        self,
        node_feats,
        out_feats,
        edge_feats,
        num_mlp_layers=2,
        hidden_mlp_dim=64,
    ):
        """
        This module implements Eq. 1 from the paper where the node features are
        updated using the neighboring node and edge features.
        Args:
            node_feats (int): Input edge feature dimension
            out_feats (int): Output feature deimension
            node_feats (int): Input node feature dimension
            num_mlp_layers (int, optional): Number of layers used in the MLP. Defaults to 2.
            hidden_mlp_dim (int, optional): Hidden feature dimension in the MLP. Defaults to 64.
        """
        super(NodeConv, self).__init__()
        self.gconv = dgl.nn.NNConv(
            in_feats=node_feats,
            out_feats=out_feats,
            edge_func=nn.Linear(edge_feats, node_feats * out_feats),
            aggregator_type="sum",
            bias=False,
        )
        self.batchnorm = nn.BatchNorm1d(out_feats)
        self.mlp = MLP(num_mlp_layers, node_feats, hidden_mlp_dim, out_feats)
        self.eps = torch.nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, graph, nfeat, efeat):
        h = (1 + self.eps) * nfeat
        h = self.gconv(graph, h, efeat)
        h = self.mlp(h)
        h = F.leaky_relu(self.batchnorm(h))
        return h


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class EdgeMPNN(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        mlp_ratio=4,
        drop=0.,
        drop_path=0.
    ):
        """
        This module implements Eq. 2 from the paper where the edge features are
        updated using the node features at the endpoints.
        Args:
            edge_feats (int): Input edge feature dimension
            out_feats (int): Output feature deimension
            node_feats (int): Input node feature dimension
            num_mlp_layers (int, optional): Number of layers used in the MLP. Defaults to 2.
            hidden_mlp_dim (int, optional): Hidden feature dimension in the MLP. Defaults to 64.
        """
        super(EdgeMPNN, self).__init__()
        # edge mpnn
        self.proj = MLP(1, node_dim, 0, edge_dim)
        self.norm_drop = nn.Sequential(nn.BatchNorm1d(edge_dim),
                                       nn.Dropout(drop))

        self.norm1 = nn.LayerNorm(edge_dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=edge_dim)
        self.res_scale1 = Scale(dim=edge_dim)

        self.norm2 = nn.LayerNorm(edge_dim)
        self.mlp = MLP(2, edge_dim, edge_dim*mlp_ratio, edge_dim, nn.LayerNorm, nn.Mish)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=edge_dim)
        self.res_scale2 = Scale(dim=edge_dim)

    def message_passing(self, g, h, he):
        with g.local_scope():
            g.ndata['h'] = h
            out_src = self.proj(g.ndata['h'])
            out_dst = self.proj(g.ndata['h'])
            g.srcdata.update({'out_src': out_src})
            g.dstdata.update({'out_dst': out_dst})
            g.apply_edges(fn.u_add_v('out_src', 'out_dst', 'out'))
            he = he + g.edata['out']
        return self.norm_drop(he)

    def forward(self, g, h, he):
        he = self.res_scale1(he) + \
            self.layer_scale1(
                self.drop_path1(
                    self.message_passing(g, h, self.norm1(he))
                )
            )
        he = self.res_scale2(he) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(he))
                )
            )
        return he


class NodeMPNN(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        delta,
        mlp_ratio=4,
        drop=0.,
        drop_path=0.
    ):
        """

        Args:
            input_dim (int): [description]
            input_edge_dim (int): [description]
            output_dim (int): [description]
            num_layers (int, optional): [description].
        """
        super(NodeMPNN, self).__init__()
        self.norm1 = nn.LayerNorm(node_dim)
        self.conv = PNAConvTower(
                        in_size=node_dim, 
                        out_size=node_dim, 
                        aggregators=['sum', 'max'], # ['sum', 'max', 'min']
                        scalers=['identity'], # ['identity', 'amplification', 'attenuation']
                        delta=delta, 
                        dropout=drop,
                        edge_feat_size=edge_dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=node_dim)
        self.res_scale1 = Scale(dim=node_dim)

        self.norm2 = nn.LayerNorm(node_dim)
        self.mlp = MLP(2, node_dim, node_dim*mlp_ratio, node_dim, nn.LayerNorm, nn.Mish)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=node_dim)
        self.res_scale2 = Scale(dim=node_dim)

    def forward(self, g, h, he):
        h = self.res_scale1(h) + \
            self.layer_scale1(
                self.drop_path1(
                    self.conv(g, self.norm1(h), he)
                )
            )
        h = self.res_scale2(h) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(h))
                )
            )
        return h
    
    
class simpleMLP(nn.Sequential):
    r"""
    Description
    -----------
    From equation (5) in "DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>"
    """

    def __init__(self, channels, act="relu", dropout=0.0, bias=True):
        layers = []

        for i in range(1, len(channels)):
            layers.append(nn.Linear(channels[i - 1], channels[i], bias))
            if i < len(channels) - 1:
                layers.append(nn.BatchNorm1d(channels[i], affine=True))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        super().__init__(*layers)


class MessageNorm(nn.Module):
    r"""
    Description
    -----------
    Message normalization was introduced in "DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>"
    Parameters
    ----------
    learn_scale: bool
        Whether s is a learnable scaling factor or not. Default is False.
    """

    def __init__(self, learn_scale=False):
        super(MessageNorm, self).__init__()
        self.scale = nn.Parameter(
            torch.FloatTensor([1.0]), requires_grad=learn_scale
        )

    def forward(self, feats, msg, p=2):
        msg = F.normalize(msg, p=2, dim=-1)
        feats_norm = feats.norm(p=p, dim=-1, keepdim=True)
        return msg * feats_norm * self.scale

    
class GENConv(nn.Module):
    r"""
    Description
    -----------
    Generalized Message Aggregator was introduced in "DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>"
    Parameters
    ----------
    in_dim: int
        Input size.
    out_dim: int
        Output size.
    aggregator: str
        Type of aggregation. Default is 'softmax'.
    beta: float
        A continuous variable called an inverse temperature. Default is 1.0.
    learn_beta: bool
        Whether beta is a learnable variable or not. Default is False.
    p: float
        Initial power for power mean aggregation. Default is 1.0.
    learn_p: bool
        Whether p is a learnable variable or not. Default is False.
    msg_norm: bool
        Whether message normalization is used. Default is False.
    learn_msg_scale: bool
        Whether s is a learnable scaling factor or not in message normalization. Default is False.
    mlp_layers: int
        The number of MLP layers. Default is 1.
    eps: float
        A small positive constant in message construction function. Default is 1e-7.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        aggregator="softmax",
        beta=1.0,
        learn_beta=False,
        p=1.0,
        learn_p=False,
        msg_norm=False,
        learn_msg_scale=False,
        mlp_layers=1,
        eps=1e-7,
    ):
        super(GENConv, self).__init__()

        self.aggr = aggregator
        self.eps = eps

        channels = [in_dim]
        for _ in range(mlp_layers - 1):
            channels.append(in_dim * 2)
        channels.append(out_dim)

        self.mlp = simpleMLP(channels)
        self.msg_norm = MessageNorm(learn_msg_scale) if msg_norm else None

        self.beta = (
            nn.Parameter(torch.Tensor([beta]), requires_grad=True)
            if learn_beta and self.aggr == "softmax"
            else beta
        )
        self.p = (
            nn.Parameter(torch.Tensor([p]), requires_grad=True)
            if learn_p
            else p
        )
        
    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            # Node and edge feature size need to match.
            g.ndata["h"] = node_feats
            g.edata["h"] = edge_feats
            g.apply_edges(fn.u_add_e("h", "h", "m"))

            if self.aggr == "softmax":
                g.edata["m"] = F.relu(g.edata["m"]) + self.eps
                g.edata["a"] = dgl.nn.functional.edge_softmax(g, g.edata["m"] * self.beta)
                g.update_all(
                    lambda edge: {"x": edge.data["m"] * edge.data["a"]},
                    fn.sum("x", "m"),
                )

            elif self.aggr == "power":
                minv, maxv = 1e-7, 1e1
                torch.clamp_(g.edata["m"], minv, maxv)
                g.update_all(
                    lambda edge: {"x": torch.pow(edge.data["m"], self.p)},
                    fn.mean("x", "m"),
                )
                torch.clamp_(g.ndata["m"], minv, maxv)
                g.ndata["m"] = torch.pow(g.ndata["m"], self.p)

            else:
                raise NotImplementedError(
                    f"Aggregator {self.aggr} is not supported."
                )

            if self.msg_norm is not None:
                g.ndata["m"] = self.msg_norm(node_feats, g.ndata["m"])

            feats = node_feats + g.ndata["m"]

            return self.mlp(feats)


class NodeMPNNV2(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        delta,
        mlp_ratio=4,
        drop=0.,
        drop_path=0.
    ):
        """

        Args:
            input_dim (int): [description]
            input_edge_dim (int): [description]
            output_dim (int): [description]
            num_layers (int, optional): [description].
        """
        super(NodeMPNNV2, self).__init__()
        self.norm1 = nn.LayerNorm(node_dim)
        self.conv = GENConv(node_dim, node_dim, learn_beta=True, learn_p=True)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=node_dim)
        self.res_scale1 = Scale(dim=node_dim)

        self.norm2 = nn.LayerNorm(node_dim)
        self.mlp = MLP(2, node_dim, node_dim*mlp_ratio, node_dim, nn.LayerNorm, nn.Mish)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=node_dim)
        self.res_scale2 = Scale(dim=node_dim)

    def forward(self, g, h, he):
        h = self.res_scale1(h) + \
            self.layer_scale1(
                self.drop_path1(
                    self.conv(g, self.norm1(h), he)
                )
            )
        h = self.res_scale2(h) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(h))
                )
            )
        return h