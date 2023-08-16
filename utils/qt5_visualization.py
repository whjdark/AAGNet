import os
import sys
import time

from OCC.Display.backend import load_backend
from OCC.Display.OCCViewer import rgb_color
from OCC.Core.AIS import AIS_ColoredShape
from OCC.Extend.TopologyUtils import TopologyExplorer
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QGridLayout,
    QVBoxLayout,
    QDialog,
    QFileDialog,
    QMessageBox,
    QListWidget,
    QListWidgetItem
)
load_backend("qt-pyqt5")
import OCC.Display.qtDisplay as qtDisplay

import numpy as np
import torch

from dataset.AAGExtractor import AAGExtractor
from dataset.topologyCheker import TopologyChecker
from models.inst_segmentors import AAGNetSegmentor
from utils.data_utils import load_one_graph, load_body_from_step, load_json_or_pkl, load_statistics
from utils.data_utils import center_and_scale, standardization



class FeatureInstance():
    def __init__(self, name:str, faces:list, bottoms:list):
        self.name = name
        self.faces = faces
        self.bottoms = bottoms


class App(QDialog):
    def __init__(self):
        super().__init__()
        # UI settings
        self.title = "Intelligent Machining Feature Recognizer based on PyQt5 & pythonOCC & DGL & Pytorch by WHJ"
        self.left = 300
        self.top = 300
        self.width = 1366
        self.height = 900
        self.canvas_width = 1000
        self.height_width = 700
        # members
        self.ais_shape = None
        self.file_name = None
        self.faces_list = []
        self.features_list = []
        # inference parameters
        self.eps = 1e-6 # small number
        self.weight_path = "./weights/weight_on_MFInstseg.pth"
        self.model_type = 'full' # ''tiny' or 'full'
        self.device = 'cuda'
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
        self.deremap = [1, 12, 14, 6, 0, 23, 24]
        self.attribute_schema = load_json_or_pkl('./feature_lists/all.json')
        self.stat = load_statistics('./weights/attr_stat.json')

        self.initRecognizer()
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setMinimumHeight(self.height // 4)
        self.setMinimumWidth(self.width // 4)
        self.createHorizontalLayout()
        self.msgBox = QMessageBox()

        windowLayout = QHBoxLayout()
        windowLayout.addWidget(self.button_panel)
        windowLayout.addWidget(self.horizontalGroupBox)
        self.setLayout(windowLayout)
        self.show()

    def initRecognizer(self):
        self.recognizer = AAGNetSegmentor(arch='AAGNetGraphEncoder',
                            num_classes=len(self.feat_names),
                            edge_attr_dim=12, node_attr_dim=10, 
                            edge_attr_emb=64, node_attr_emb=64,
                            edge_grid_dim=0, node_grid_dim=7,
                            edge_grid_emb=0, node_grid_emb=64,
                            num_layers=3, delta=2, mlp_ratio=2,
                            drop=0., drop_path=0.,
                            head_hidden_dim=64,
                            conv_on_edge=False)
        model_param = torch.load(self.weight_path, map_location='cpu')
        self.recognizer.load_state_dict(model_param)
        self.recognizer = self.recognizer.to(self.device)
        self.recognizer.eval()

    def createHorizontalLayout(self):
        self.horizontalGroupBox = QWidget()
        layout = QHBoxLayout()
        self.button_panel = QWidget()
        self.button_panel.setFixedWidth(200) 
        panel_layout = QGridLayout()

        disp = QPushButton("load STEP", self)
        disp.clicked.connect(self.openShape)
        panel_layout.addWidget(disp, 0, 0, 1, 1)

        eras = QPushButton("Close STEP", self)
        eras.clicked.connect(self.eraseShape)
        panel_layout.addWidget(eras, 1, 0, 1, 1)

        feature_rec = QPushButton("Feature Recognition", self)
        feature_rec.clicked.connect(self.featureRecog)
        panel_layout.addWidget(feature_rec, 2, 0, 1, 1)

        self.canvas = qtDisplay.qtViewer3d(self)
        layout.addWidget(self.canvas)
        self.canvas.resize(self.canvas_width, self.height_width)
        self.canvas.InitDriver()
        self.display = self.canvas._display

        self.featureListWidget = QListWidget()
        self.featureListWidget.itemDoubleClicked.connect(self.featureListDoubleClicked)
        panel_layout.addWidget(self.featureListWidget, 3, 0, 1, 1)

        self.horizontalGroupBox.setLayout(layout)
        self.button_panel.setLayout(panel_layout)

    def centroid_attribute(self, face):
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
        mass_props = GProp_GProps()
        brepgprop_SurfaceProperties(face, mass_props)
        gPt = mass_props.CentreOfMass()

        return gPt.Coord()

    def openShape(self):
        self.file_name = QFileDialog.getOpenFileName(self, "Open Step File", "./", "(*.st*p)")[0]
        if self.file_name is None or self.file_name == '':
            return
        solid = load_body_from_step(self.file_name)
        if not self.topoChecker(solid):
            self.msgBox.warning(self, "warning", "Fail to load, unsupported or wrong STEP.")
            self.file_name = None
            return
        # clear canvas
        self.display.Context.Erase(self.ais_shape, True)

        # reset feature storage variables
        self.featureListWidget.clear()
        self.features_list.clear()

        # show the colored shape
        self.ais_shape = AIS_ColoredShape(solid)
        self.display.Context.Display(self.ais_shape, True)
        self.display.FitAll()

        # read the faces from shape
        topo = TopologyExplorer(solid)
        self.faces_list = list(topo.faces())

    def eraseShape(self):
        if self.ais_shape:
            self.display.Context.Erase(self.ais_shape, True)
            self.ais_shape = None
            self.file_name = None
            self.featureListWidget.clear()
            self.features_list.clear()
            self.faces_list.clear()

    def featureRecog(self):
        # Check if the ais_shape and file_name variables are set
        if self.ais_shape and self.file_name:
            start_time = time.time()
            # Extract the attributes adjacency graph from the 3D shape
            try:
                aagExt = AAGExtractor(self.file_name, self.attribute_schema)
                aag = aagExt.process()
            except Exception as e:
                self.msgBox.warning(self, 
                    "warning", 'Wrong {} with Exception: {}'.format(e, self.file_name)) 
                return

            # Convert the adjacency graph to a tensor
            sample = load_one_graph(self.file_name, aag)
            if self.normalize:
                one_graph = standardization(sample, self.stat)
            if self.center_and_scale:
                sample = center_and_scale(sample)
            one_graph = sample["graph"]
            one_graph = one_graph.to(self.device)
            #print(tensor_aag.ndata["grid"])
            pre_time = time.time()
            print(f'Pre-processing duration: {pre_time - start_time}')

            with torch.no_grad():
                # Use the pre-trained model to make predictions on the tensor
                try:
                    seg_out, inst_out, bottom_out = self.recognizer(one_graph)
                except Exception as e:
                    self.msgBox.warning(self, 
                        "warning", 'Inference failed with Exception: {}'.format(e)) 
                    return
                
                # Unpack the model's output
                ff_time = time.time()
                print(f'Feed-forward duration: {ff_time - pre_time}')

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
                used_flags = np.zeros(adj.shape[0], dtype=np.bool8)
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
                
                for row, proposal in enumerate(adj.tolist()):
                    print(row, proposal)
                print(f'This Shape has {len(self.faces_list)} faces')
                print(f'This Shape has {len(proposals)} proposals')

                # clear previous results
                self.featureListWidget.clear()
                self.features_list.clear()
                # save new results
                print(proposals)
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

                    if inst_logit == 1 and len(instance) > 1:
                        # check if every face in this instance is same direction
                        # if not, ignore
                        dirs = [self.centroid_attribute(self.faces_list[face]) for face in instance]
                        # if dir differ < eps, they are same through hole
                        # else, they are not and should be considered as another feature
                        res = (np.linalg.norm(np.array(dirs) - dirs[0], axis=1) > self.eps)
                        if np.sum(res) > 0:
                            # remove this face from the instance, they are not same through hole
                            new_instance = [[instance[i]] for i in range(len(instance)) if res[i] == True]
                            instance = [instance[i] for i in range(len(instance)) if res[i] == False]
                            inst_name = self.feat_names[inst_logit] 
                            new_instance.append(instance) 
                            for inst in new_instance:
                                self.features_list.append(
                                    FeatureInstance(name=inst_name, faces=inst, bottoms=[]))
                                self.featureListWidget.addItem(inst_name)
                            continue

                    # get instance label name from face_logits
                    if self.model_type == 'tiny':
                        inst_logit = self.deremap[inst_logit]
                    inst_name = self.feat_names[inst_logit] 
                    print(inst_name, sum_inst_logit)
                    # get the bottom faces
                    bottom_faces = []
                    for face_idx in instance:
                        if bottom_logits[face_idx]:
                            bottom_faces.append(face_idx)
                    self.features_list.append(
                        FeatureInstance(name=inst_name, faces=instance, bottoms=bottom_faces))
                    self.featureListWidget.addItem(inst_name)
                
                post_time = time.time()
                print(f'Post-processing duration: {post_time - ff_time}')

            print(f'Total duration: {time.time() - start_time}s')

    def featureListDoubleClicked(self):
        if self.ais_shape and self.file_name and len(self.features_list) != 0:
            selected_row = self.featureListWidget.currentRow()
            selected_inst = self.features_list[selected_row]
            print(selected_inst.name, selected_inst.faces)

            # reset the color of solid faces
            self.ais_shape.ClearCustomAspects()
            # set the color of selected feature faces
            for face in self.faces_list:
                self.ais_shape.SetCustomColor(face, rgb_color(1,1,1))
                self.ais_shape.SetCustomTransparency(face, 0.6)

            for face_idx in selected_inst.faces:
                face = self.faces_list[face_idx]
                self.ais_shape.SetCustomColor(face, rgb_color(0,1,0))
                self.ais_shape.SetCustomTransparency(face, 0.0)

            if len(selected_inst.bottoms) > 0:
                for face_idx in selected_inst.bottoms:
                    face = self.faces_list[face_idx]
                    self.ais_shape.SetCustomColor(face, rgb_color(1,0,0))
                    self.ais_shape.SetCustomTransparency(face, 0.0)

            self.display.Context.Display(self.ais_shape, True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    if os.getenv("APPVEYOR") is None:
        sys.exit(app.exec_())