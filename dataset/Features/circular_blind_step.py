import numpy as np

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.gp import gp_Circ, gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.GC import GC_MakeSegment, GC_MakeArcOfCircle
import Utils.occ_utils as occ_utils

from Features.machining_features import MachiningFeature


class CircularBlindStep(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = 2
        self.bound_type = 2
        self.depth_type = "blind"
        self.feat_type = "circular_blind_step"

    def _add_sketch(self, bound):
        vec0 = bound[0] - bound[1]
        vec2 = bound[2] - bound[1]
        width = np.linalg.norm(vec2)
        height = np.linalg.norm(vec0)

        radius = min(width, height)
        vec0 = vec0 / np.linalg.norm(vec0)
        vec2 = vec2 / np.linalg.norm(vec2)

        pt0 = occ_utils.as_occ(bound[1] + vec0 * radius, gp_Pnt)
        pt1 = occ_utils.as_occ(bound[1], gp_Pnt)
        pt2 = occ_utils.as_occ(bound[1] + vec2 * radius, gp_Pnt)

        normal = occ_utils.as_occ(np.cross(vec2, vec0), gp_Dir)
        cir = gp_Circ(gp_Ax2(pt1, normal), radius)
        seg_maker = [GC_MakeSegment(pt0, pt1), GC_MakeSegment(pt1, pt2), GC_MakeArcOfCircle(cir, pt2, pt0, True)]
        wire_maker = BRepBuilderAPI_MakeWire()
        for sm in seg_maker:
            if sm.IsDone:
                edge = BRepBuilderAPI_MakeEdge(sm.Value()).Edge()
                wire_maker.Add(edge)
            else:
                return None

        face_maker = BRepBuilderAPI_MakeFace(wire_maker.Wire())

        return face_maker.Face()