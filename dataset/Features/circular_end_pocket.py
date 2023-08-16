import numpy as np

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.gp import gp_Circ, gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.GC import GC_MakeSegment, GC_MakeArcOfCircle
import Utils.occ_utils as occ_utils

from Features.machining_features import MachiningFeature


class CircularEndPocket(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = 4
        self.bound_type = 4
        self.depth_type = "blind"
        self.feat_type = "circular_end_pocket"

    def _add_sketch(self, bound):
        dir_w = bound[2] - bound[1]
        dir_h = bound[0] - bound[1]
        width = np.linalg.norm(dir_w)
        height = np.linalg.norm(dir_h)

        pt0 = bound[0]
        pt1 = bound[1]
        pt2 = bound[2]
        pt3 = bound[3]
        if width < height:
            pt0 = bound[1]
            pt1 = bound[2]
            pt2 = bound[3]
            pt3 = bound[0]
            radius = width
            width = height
            height = radius

        radius = min(height / 2, width / 2 - 0.5)
        dir_w = (pt2 - pt1) / width
        dir_h = (pt0 - pt1) / height

        pt1 = pt1 + dir_w * radius
        pt2 = pt2 - dir_w * radius
        pt3 = pt2 + dir_h * 2 * radius  # TODO(ANDREW): Why does this not use pt3?
        pt0 = pt1 + dir_h * 2 * radius
        c01 = (pt1 + pt0) / 2
        c23 = (pt2 + pt3) / 2

        pt1 = occ_utils.as_occ(pt1, gp_Pnt)
        pt0 = occ_utils.as_occ(pt0, gp_Pnt)
        c01 = occ_utils.as_occ(c01, gp_Pnt)
        pt2 = occ_utils.as_occ(pt2, gp_Pnt)
        pt3 = occ_utils.as_occ(pt3, gp_Pnt)
        c23 = occ_utils.as_occ(c23, gp_Pnt)

        normal = occ_utils.as_occ(np.cross(dir_w, dir_h), gp_Dir)
        cir01 = gp_Circ(gp_Ax2(c01, normal), radius)
        cir23 = gp_Circ(gp_Ax2(c23, normal), radius)
        seg_maker = [GC_MakeArcOfCircle(cir01, pt0, pt1, True), GC_MakeSegment(pt1, pt2),
                     GC_MakeArcOfCircle(cir23, pt2, pt3, True), GC_MakeSegment(pt3, pt0)]
        wire_maker = BRepBuilderAPI_MakeWire()
        for sm in seg_maker:
            if sm.IsDone:
                edge = BRepBuilderAPI_MakeEdge(sm.Value()).Edge()
                wire_maker.Add(edge)
            else:
                return None

        face_maker = BRepBuilderAPI_MakeFace(wire_maker.Wire())

        return face_maker.Face()