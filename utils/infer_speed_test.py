import os
import sys
import time

import numpy as np
import torch

from dataset.AAGExtractor import AAGExtractor
from dataset.topologyCheker import TopologyChecker
from models.inst_segmentors import AAGNetSegmentor
from utils.data_utils import load_one_graph, load_body_from_step, load_json_or_pkl, load_statistics
from utils.data_utils import center_and_scale, standardization
from utils.misc import print_num_params

class FeatureInstance():
    def __init__(self, name:str, faces:list, bottoms:list):
        self.name = name
        self.faces = faces
        self.bottoms = bottoms

class Recognizer():
    def __init__(self, device='cuda'):
        super().__init__()
        # inference parameters
        self.eps = 1e-6 # small number
        self.weight_path = "./weights/weight_on_MFInstseg.pth"
        self.model_type = 'full' # ''tiny' or 'full'
        self.device = device
        self.center_and_scale = False
        self.normalize = True
        self.inst_thres = 0.5
        self.bottom_thres = 0.5
        self.topoChecker = TopologyChecker()
        self.feat_names = ['chamfer', 'through_hole', 'triangular_passage', 'rectangular_passage', '6sides_passage',
                           'triangular_through_slot', 'rectangular_through_slot', 'circular_through_slot',
                           'rectangular_through_step', '2sides_through_step', 'slanted_through_step', 'Oring', 'blind_hole',
                           'triangular_pocket', 'rectangular_pocket', '6sides_pocket', 'circular_end_pocket',
                           'rectangular_blind_slot', 'v_circular_end_blind_slot', 'h_circular_end_blind_slot',
                           'triangular_blind_step', 'circular_blind_step', 'rectangular_blind_step', 'round', 'stock'
                          ]
        self.config={
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
                    }
        self.n_classes = 25
        self.deremap = [1, 12, 14, 6, 0, 23, 24]
        self.attribute_schema = load_json_or_pkl('./feature_lists/all.json')
        self.stat = load_statistics('./weights/attr_stat.json')

        self.initRecognizer()

    def initRecognizer(self):
        self.recognizer = AAGNetSegmentor(num_classes=self.n_classes,
                            arch=self.config['architecture'],
                            edge_attr_dim=self.config['edge_attr_dim'], 
                            node_attr_dim=self.config['node_attr_dim'], 
                            edge_attr_emb=self.config['edge_attr_emb'], 
                            node_attr_emb=self.config['node_attr_emb'],
                            edge_grid_dim=self.config['edge_grid_dim'], 
                            node_grid_dim=self.config['node_grid_dim'], 
                            edge_grid_emb=self.config['edge_grid_emb'], 
                            node_grid_emb=self.config['node_grid_emb'], 
                            num_layers=self.config['num_layers'], 
                            delta=self.config['delta'], 
                            mlp_ratio=self.config['mlp_ratio'], 
                            drop=self.config['drop'], 
                            drop_path=self.config['drop_path'], 
                            head_hidden_dim=self.config['head_hidden_dim'],
                            conv_on_edge=self.config['conv_on_edge'],
                            use_uv_gird=self.config['use_uv_gird'],
                            use_edge_attr=self.config['use_edge_attr'],
                            use_face_attr=self.config['use_face_attr'],)
        self.recognizer = self.recognizer.to(self.device)
        self.recognizer.eval()
        print_num_params(self.recognizer)

    def featureRecog(self, file_name):
        # Extract the attributes adjacency graph from the 3D shape
        try:
            aagExt = AAGExtractor(file_name, self.attribute_schema)
            aag = aagExt.process()
        except Exception as e:
            print('Wrong {} with Exception: {}'.format(e, file_name)) 
            return

        # Convert the adjacency graph to a tensor
        sample = load_one_graph(file_name, aag)
        if self.normalize:
            one_graph = standardization(sample, self.stat)
        if self.center_and_scale:
            sample = center_and_scale(sample)
        one_graph = sample["graph"]
        one_graph = one_graph.to(self.device)
        #print(tensor_aag.ndata["grid"])

        with torch.no_grad():
            # Use the pre-trained model to make predictions on the tensor
            try:
                seg_out, inst_out, bottom_out = self.recognizer(one_graph)
            except Exception as e:
                print('Inference failed with Exception: {}'.format(e)) 
                return
        
            # post-processing for semantic segmentation 
            # face_logits = torch.argmax(seg_out, dim=1)
            face_logits = seg_out.cpu().numpy()

            # post-processing for instance segmentation 
            inst_out = inst_out[0] # inst_out is a list
            inst_out = inst_out.sigmoid()
            adj = inst_out > self.inst_thres
            adj = adj.cpu().numpy().astype('int32')

            # post-processing for bottom classification 
            bottom_out = bottom_out.sigmoid()
            bottom_logits = bottom_out > self.bottom_thres
            bottom_logits = bottom_logits.cpu().numpy()
            
            # Identify individual proposals of each feature
            proposals = set() # use to delete repeat proposals
            # record whether the face belongs to a instance
            used_flags = np.zeros(adj.shape[0], dtype=np.bool_)
            for row_idx, row in enumerate(adj):
                if used_flags[row_idx]:
                    # the face has been assigned to a instance
                    continue
                if np.sum(row) <= self.eps: 
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
            
            results = []
            # save new results
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
                if self.model_type == 'tiny':
                    inst_logit = self.deremap[inst_logit]
                inst_name = self.feat_names[inst_logit] 
                # get the bottom faces
                bottom_faces = []
                for face_idx in instance:
                    if bottom_logits[face_idx]:
                        bottom_faces.append(face_idx)
                results.append(FeatureInstance(inst_name, instance, bottom_faces))
        
        return results


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high") # may be faster if GPU support TF32
    # time test on CPU
    rec = Recognizer(device='cpu')

    # warm up
    for i in range(10):
        rec.featureRecog('./examples/partA.step')

    # time test on part A B C
    # part A
    start = time.time()
    for i in range(100):
        rec.featureRecog('./examples/partA.step')
    end = time.time()
    print('part A CPU time: {}'.format((end-start)/100))

    # part B
    start = time.time()
    for i in range(100):
        rec.featureRecog('./examples/partB.step')
    end = time.time()
    print('part B CPU time: {}'.format((end-start)/100))

    # part C
    start = time.time()
    for i in range(100):
        rec.featureRecog('./examples/partC.step')
    end = time.time()
    print('part C CPU time: {}'.format((end-start)/100))

    # time test on GPU
    rec = Recognizer(device='cuda')

    # warm up
    for i in range(10):
        rec.featureRecog('./examples/partA.step')   
    
    # time test on part A B C
    # part A
    start = time.time()
    for i in range(100):
        rec.featureRecog('./examples/partA.step')
    end = time.time()
    print('part A GPU time: {}'.format((end-start)/100))

    # part B
    start = time.time()
    for i in range(100):
        rec.featureRecog('./examples/partB.step')
    end = time.time()
    print('part B GPU time: {}'.format((end-start)/100))

    # part C
    start = time.time()
    for i in range(100):
        rec.featureRecog('./examples/partC.step')
    end = time.time()
    print('part C GPU time: {}'.format((end-start)/100))