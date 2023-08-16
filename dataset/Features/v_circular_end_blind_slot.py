import random
import numpy as np

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.gp import gp_Circ, gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.GC import GC_MakeSegment, GC_MakeArcOfCircle
import Utils.occ_utils as occ_utils

from Features.machining_features import MachiningFeature


class VCircularEndBlindSlot(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = 1
        self.bound_type = 1
        self.depth_type = "blind"
        self.feat_type = "v_circular_end_blind_slot"

    def _add_sketch(self, bound):
        dir_w = bound[2] - bound[1]
        dir_h = bound[0] - bound[1]
        width = np.linalg.norm(dir_w)
        height = np.linalg.norm(dir_h)
        if height - width / 2 > 1.0:
            radius = width / 2
        else:
            radius = random.uniform(0.5, height / 2)

        offset = width / 2 - radius
        dir_w = dir_w / width
        dir_h = dir_h / height
        normal = np.cross(dir_w, dir_h)

        pt1 = bound[1] + dir_w * offset
        pt2 = pt1 + dir_w * 2 * radius
        pt3 = pt2 + dir_h * (height - radius)
        pt4 = pt1 + dir_h * (height - radius)

        center = pt4 + dir_w * radius
        circ = gp_Circ(gp_Ax2(gp_Pnt(center[0], center[1], center[2]), occ_utils.as_occ(normal, gp_Dir)), radius)
        pt1 = occ_utils.as_occ(pt1, gp_Pnt)
        pt2 = occ_utils.as_occ(pt2, gp_Pnt)
        pt3 = occ_utils.as_occ(pt3, gp_Pnt)
        pt4 = occ_utils.as_occ(pt4, gp_Pnt)
        seg_maker = [GC_MakeSegment(pt1, pt2), GC_MakeSegment(pt2, pt3), GC_MakeArcOfCircle(circ, pt3, pt4, True),
                     GC_MakeSegment(pt4, pt1)]
        wire_maker = BRepBuilderAPI_MakeWire()
        for sm in seg_maker:
            edge = BRepBuilderAPI_MakeEdge(sm.Value()).Edge()
            wire_maker.Add(edge)

        face_maker = BRepBuilderAPI_MakeFace(wire_maker.Wire())

        return face_maker.Face()