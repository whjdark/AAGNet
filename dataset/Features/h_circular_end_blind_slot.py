import numpy as np

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.gp import gp_Circ, gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.GC import GC_MakeSegment, GC_MakeArcOfCircle
import Utils.occ_utils as occ_utils

from Features.machining_features import MachiningFeature


class HCircularEndBlindSlot(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = 1
        self.bound_type = 1
        self.depth_type = "blind"
        self.feat_type = "h_circular_end_blind_slot"

    def _add_sketch(self, bound):
        dir_w = bound[2] - bound[1]
        dir_h = bound[0] - bound[1]
        width = np.linalg.norm(dir_w)
        height = np.linalg.norm(dir_h)

        dir_w = dir_w / width
        dir_h = dir_h / height
        normal = np.cross(dir_w, dir_h)

        rect_h = min(height, width / 2 - 0.5)
        pt1 = bound[1]
        pt2 = bound[2]
        pt3 = pt2 - dir_w * rect_h + dir_h * rect_h
        pt4 = pt1 + dir_w * rect_h + dir_h * rect_h

        center1 = pt1 + dir_w * rect_h
        center2 = pt2 - dir_w * rect_h
        circ1 = gp_Circ(gp_Ax2(occ_utils.as_occ(center1, gp_Pnt), occ_utils.as_occ(normal, gp_Dir)), rect_h)
        circ2 = gp_Circ(gp_Ax2(occ_utils.as_occ(center2, gp_Pnt), occ_utils.as_occ(normal, gp_Dir)), rect_h)
        pt1 = occ_utils.as_occ(pt1, gp_Pnt)
        pt2 = occ_utils.as_occ(pt2, gp_Pnt)
        pt3 = occ_utils.as_occ(pt3, gp_Pnt)
        pt4 = occ_utils.as_occ(pt4, gp_Pnt)

        seg_maker = [GC_MakeSegment(pt1, pt2), GC_MakeArcOfCircle(circ2, pt2, pt3, True), GC_MakeSegment(pt3, pt4),
                     GC_MakeArcOfCircle(circ1, pt4, pt1, True)]
        wire_maker = BRepBuilderAPI_MakeWire()
        for sm in seg_maker:
            edge = BRepBuilderAPI_MakeEdge(sm.Value()).Edge()
            wire_maker.Add(edge)

        face_maker = BRepBuilderAPI_MakeFace(wire_maker.Wire())

        return face_maker.Face()