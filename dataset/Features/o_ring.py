import random
import math
import numpy as np

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.gp import gp_Circ, gp_Ax2, gp_Pnt, gp_Dir

import Utils.occ_utils as occ_utils
from Features.machining_features import MachiningFeature


class ORing(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = 4
        self.bound_type = 4
        self.depth_type = "blind"
        self.feat_type = "Oring"

    def _add_sketch(self, bound):
        dir_w = bound[2] - bound[1]
        dir_h = bound[0] - bound[1]
        width = np.linalg.norm(dir_w)
        height = np.linalg.norm(dir_h)

        dir_w = dir_w / width
        dir_h = dir_h / height
        normal = occ_utils.as_occ(np.cross(dir_w, dir_h), gp_Dir)

        outer_r = min(width / 2, height / 2)
        center = (bound[0] + bound[1] + bound[2] + bound[3]) / 4

        inner_r = random.uniform(outer_r / 3, outer_r - 0.2)

        circ = gp_Circ(gp_Ax2(gp_Pnt(center[0], center[1], center[2]), normal), outer_r)
        edge = BRepBuilderAPI_MakeEdge(circ, 0., 2 * math.pi).Edge()
        outer_wire = BRepBuilderAPI_MakeWire(edge).Wire()

        normal.Reverse()
        circ = gp_Circ(gp_Ax2(gp_Pnt(center[0], center[1], center[2]), normal), inner_r)
        edge = BRepBuilderAPI_MakeEdge(circ, 0., 2 * math.pi).Edge()
        inner_wire = BRepBuilderAPI_MakeWire(edge).Wire()

        face_maker = BRepBuilderAPI_MakeFace(outer_wire)
        face_maker.Add(inner_wire)

        return face_maker.Face()
