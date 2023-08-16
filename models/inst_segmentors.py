import torch
from torch import nn

import models.encoders as encoders
from .decoders import InnerProductDecoder, InstanceDecoder
from .layers import NonLinearClassifier, MLP



###############################################################################
# Segmentation model
###############################################################################

class AAGNetSegmentor(nn.Module):
    """
    AAG-Net solid face segmentation model
    """

    def __init__(self,
                 num_classes,
                 arch,
                 edge_attr_dim,
                 node_attr_dim,
                 edge_attr_emb,
                 node_attr_emb,
                 edge_grid_dim,
                 node_grid_dim,
                 edge_grid_emb,
                 node_grid_emb,
                 num_layers,
                 delta,
                 mlp_ratio=4,
                 drop=0., 
                 drop_path=0.,
                 head_hidden_dim=256,
                 conv_on_edge=True,
                 use_uv_gird=True,
                 use_edge_attr=True,
                 use_face_attr=True,
                ):
        """
        Initialize the AAG-Net solid face segmentation model

        Args:
            num_classes (int): Number of classes to output per-face
            crv_in_channels (int, optional): Number of input channels for the 1D edge UV-grids
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the whole graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()
        # A linear network to encode B-rep face attributes
        self.use_uv_gird = use_uv_gird
        self.use_edge_attr = use_edge_attr
        self.use_face_attr = use_face_attr
        self.node_attr_encoder = nn.Sequential(
                                    nn.Linear(node_attr_dim, node_attr_emb),
                                    nn.LayerNorm(node_attr_emb))
        self.node_grid_encoder = None
        if node_grid_dim:
            self.node_grid_encoder = nn.Sequential(nn.Conv2d(node_grid_dim, node_grid_emb // 4, 3, 1, 1),
                                                   nn.BatchNorm2d(node_grid_emb // 4),
                                                   nn.Mish(),
                                                   nn.Conv2d(node_grid_emb // 4, node_grid_emb // 2, 3, 1, 1),
                                                   nn.BatchNorm2d(node_grid_emb // 2),
                                                   nn.Mish(),
                                                   nn.Conv2d(node_grid_emb // 2, node_grid_emb, 3, 1, 1),
                                                   nn.BatchNorm2d(node_grid_emb),
                                                   nn.Mish(),
                                                   nn.AdaptiveAvgPool2d(1),
                                                   nn.Flatten(1))
        # A linear network to encode B-rep edge attributes
        self.edge_attr_encoder = nn.Sequential(nn.Linear(edge_attr_dim, edge_attr_emb),
                                               nn.LayerNorm(edge_attr_emb))
        if edge_grid_dim:
            # TODO
            pass
        node_emb = node_attr_emb + node_grid_emb
        edge_emb = edge_attr_emb + edge_grid_emb
        # A graph neural network that message passes face and edge features
        encoder = getattr(encoders, arch)
        self.graph_encoder = encoder(node_emb, edge_emb, 
                                     num_layers, 
                                     delta, mlp_ratio, 
                                     drop, drop_path,
                                     conv_on_edge)
        final_out_emb = 2 * node_emb
        # A non-linear classifier that maps face embeddings to face logits
        self.seg_head = MLP(num_layers=2,
                            input_dim=final_out_emb, 
                            hidden_dim=head_hidden_dim,
                            output_dim=num_classes,
                            norm=nn.LayerNorm,
                            act=nn.Mish)
        # An InnerProductDecoder that maps face embeddings to face adjacency graph
        Wk = MLP(num_layers=2,
                 input_dim=final_out_emb, 
                 hidden_dim=head_hidden_dim,
                 output_dim=head_hidden_dim,
                 norm=nn.LayerNorm,
                 last_norm=True,
                 act=nn.Mish)
        Wq = MLP(num_layers=2,
                 input_dim=final_out_emb, 
                 hidden_dim=head_hidden_dim,
                 output_dim=head_hidden_dim,
                 norm=nn.LayerNorm,
                 last_norm=True,
                 act=nn.Mish)
        self.inst_head = InnerProductDecoder(Wq, Wk)
        # A non-linear classifier that maps face embeddings to face bottom logits (binary classification)
        self.bottom_head = MLP(num_layers=2,
                               input_dim=final_out_emb, 
                               hidden_dim=head_hidden_dim,
                               output_dim=1,
                               norm=nn.LayerNorm,
                               act=nn.Mish)

    def forward(self, batched_graph):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: 
                Logits (total_nodes_in_batch x num_classes)
                Bottom Logits (total_nodes_in_batch x 1)
            list [torch.tensor]:
                Face adjacency graph (num_graph_per_batch, num_faces x num_faces)
        """
        # Input features
        input_node_attr = batched_graph.ndata["x"] if self.use_face_attr else torch.zeros_like(batched_graph.ndata["x"])
        input_node_grid = batched_graph.ndata["grid"] if self.use_uv_gird else torch.zeros_like(batched_graph.ndata["grid"])
        input_edge_attr = batched_graph.edata["x"] if self.use_edge_attr else torch.zeros_like(batched_graph.edata["x"])
        # input_edge_grid = batched_graph.edata["grid"]
        # Compute hidden face features
        node_feat = self.node_attr_encoder(input_node_attr)
        if self.node_grid_encoder:
            assert input_node_grid.numel() > 0
            node_grid_feat = self.node_grid_encoder(input_node_grid)
            node_feat = torch.concat([node_feat, node_grid_feat], dim=1)
        # Compute hidden edge features
        edge_feat = self.edge_attr_encoder(input_edge_attr)
        # Message pass and compute per-face(node) and global embeddings
        node_emb, graph_emb = self.graph_encoder(
            batched_graph, node_feat, edge_feat
        )
        # concatenated to the per-node embeddings
        num_nodes_per_graph = batched_graph.batch_num_nodes().to(graph_emb.device)
        graph_emb = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
        local_global_feat = torch.cat((node_emb, graph_emb), dim=1)
        # Map to logits
        seg_out = self.seg_head(local_global_feat)
        inst_out, feat_list = self.inst_head(batched_graph, local_global_feat)
        bottom_out = self.bottom_head(local_global_feat)

        return seg_out, inst_out, bottom_out