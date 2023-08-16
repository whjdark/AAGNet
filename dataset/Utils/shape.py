import os
import pickle
import numpy as np
import csv

from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.STEPConstruct import stepconstruct_FindEntity
from OCC.Core.TCollection import TCollection_HAsciiString
from OCC.Core.StepRepr import StepRepr_RepresentationItem
from OCC.Core.AIS import AIS_ColoredShape
from OCC.Display.OCCViewer import rgb_color
from OCC.Core.TopAbs import TopAbs_FACE

import Utils.occ_utils as occ_utils
#import Utils.feature as feature
import Utils.parameters as param
import feature_creation


def shape_with_fid_to_step(filename, shape, id_map):
    """Save shape to a STEP file format.

    :param filename: Name to save shape as.
    :param shape: Shape to be saved.
    :param id_map: Variable mapping labels to faces in shape.
    :return: None
    """

    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)

    finderp = writer.WS().TransferWriter().FinderProcess()

    fset = occ_utils.list_face(shape)

    loc = TopLoc_Location()
    for face in fset:
        item = stepconstruct_FindEntity(finderp, face, loc)
        if item is None:
            print(face)
            continue
        item.SetName(TCollection_HAsciiString(str(id_map[face])))

    writer.Write(filename)


def shape_with_fid_from_step(filename):
    """Read shape from STEP file

    :param filename: Name of STEP file
    :return: shape -> TopoDS_shape, id_map -> {TopoDS_Face: int}
    """

    if not os.path.exists(filename):
        print(filename, ' not exists')
        return

    reader = STEPControl_Reader()
    reader.ReadFile(filename)
    reader.TransferRoots()
    shape = reader.OneShape()

    treader = reader.WS().TransferReader()

    id_map = {}
    fset = occ_utils.list_face(shape)
    # read the face names
    for face in fset:
        item = treader.EntityFromShapeResult(face, 1)
        if item is None:
            print(face)
            continue
        item = StepRepr_RepresentationItem.DownCast(item)
        name = item.Name().ToCString()
        if name:
            nameid = int(name)
            id_map[face] = nameid

    return shape, id_map


class LabeledShape:
    def __init__(self):
        self.shape_name = ''
        self.face_ids = {}
        self.face_truth = []

    def directive(self, combo, count):
        self.shape, face_label_map = feature_creation.shape_from_directive(combo)

        cnt = 0
        for face in face_label_map:
            self.face_ids[face] = cnt
            cnt += 1
            self.face_truth.append(face_label_map[face])

        """
        self.shape_name = ''
        for i in range(len(self.final_comb)):
            if i < len(self.final_comb) - 1:
                self.shape_name += str(self.final_comb[i]) + '-'
            else:
                self.shape_name += str(self.final_comb[i])        
        """

        self.shape_name = str(count)

    def load(self, shape_path, shape_name):
        filename = os.path.join(shape_path, shape_name + '.step')
        self.shape, self.face_ids = shape_with_fid_from_step(filename)

        filename = os.path.join(shape_path, shape_name + '.face_truth')
        with open(filename, 'rb') as file:
            self.face_truth = pickle.load(file)
        self.shape_name = shape_name

    def save(self, shape_path):
        print(f"Saving: {self.shape_name}")
        filename = os.path.join(shape_path, self.shape_name + '.step')
        shape_with_fid_to_step(filename, self.shape, self.face_ids)

        filename = os.path.join(shape_path, self.shape_name + '.face_truth')
        with open(filename, 'wb') as file:
            pickle.dump(self.face_truth, file)

    def recognize_clicked(self, shp, *kwargs):
        """ This is the function called every time
        a face is clicked in the 3d view
        """
        for shape in shp:  # this should be a TopoDS_Face TODO check it is
            if shape.ShapeType() == TopAbs_FACE:
                print(f"Face: {self.face_ids[shape]}, Class: {self.face_truth[self.face_ids[shape]]} "
                      f"-> {param.feat_names[self.face_truth[self.face_ids[shape]]]}")

    def display(self, occ_display):
        colors = []
        #rgb_list = np.array(np.meshgrid([0.0, 1.0, 1.0], [0.9, 0.6, 0.3], [0.9, 0.6, 0.3])).T.reshape(-1, 3)
        rgb_list = np.array(np.meshgrid([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])).T.reshape(-1, 3)
        for rgb in rgb_list:
            colors.append(rgb_color(rgb[0], rgb[1], rgb[2]))

        occ_display.EraseAll()
        AIS = AIS_ColoredShape(self.shape)
        face_label_map = {f: self.face_truth[self.face_ids[f]] for f in self.face_ids}
        for a_face in face_label_map:
            AIS.SetCustomColor(a_face, colors[face_label_map[a_face]])

        occ_display.register_select_callback(self.recognize_clicked)
        occ_display.Context.Display(AIS, True)
        occ_display.View_Iso()
        occ_display.FitAll()
        print(self.shape_name)
        print(self.face_truth)
        print(np.unique(self.face_truth))

