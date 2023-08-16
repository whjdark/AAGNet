import numpy as np
import Utils.occ_utils as occ_utils

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.gp import gp_Circ, gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.GC import GC_MakeSegment, GC_MakeArcOfCircle

from Features.machining_features import MachiningFeature


class CircularThroughSlot(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = 1
        self.bound_type = 1
        self.depth_type = "through"
        self.feat_type = "circular_through_slot"

    def _add_sketch(self, bound):
        # sample radius
        edge_dir = bound[2] - bound[1]
        width = np.linalg.norm(edge_dir)
        edge_dir = edge_dir / width
        height = np.linalg.norm(bound[0] - bound[1])

        radius = min(width / 2, height)
        center = (bound[1] + bound[2]) / 2
        if radius < width / 2:
            pnt1 = occ_utils.as_occ(center - edge_dir * radius, gp_Pnt)
            pnt2 = occ_utils.as_occ(center + edge_dir * radius, gp_Pnt)
        else:
            pnt1 = occ_utils.as_occ(bound[1], gp_Pnt)
            pnt2 = occ_utils.as_occ(bound[2], gp_Pnt)
        center = occ_utils.as_occ(center, gp_Pnt)

        edge1 = BRepBuilderAPI_MakeEdge(GC_MakeSegment(pnt1, pnt2).Value()).Edge()

        normal = occ_utils.as_occ(np.cross(edge_dir, bound[0] - bound[1]), gp_Dir)
        circ = gp_Circ(gp_Ax2(center, normal), radius)
        edge2 = BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(circ, pnt2, pnt1, True).Value()).Edge()

        wire_maker = BRepBuilderAPI_MakeWire(edge1, edge2)
        try:
            face_maker = BRepBuilderAPI_MakeFace(wire_maker.Wire())
        except RuntimeError as error:
            print(error)
            return None
        return face_maker.Face()