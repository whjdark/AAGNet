import numpy as np
import Utils.occ_utils as occ_utils

from Features.machining_features import MachiningFeature


class TriangularThroughSlot(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = 1
        self.bound_type = 1
        self.depth_type = "through"
        self.feat_type = "triangular_through_slot"

    def _add_sketch(self, bound):
        edge_dir = bound[2] - bound[1]

        normal = np.cross(edge_dir, bound[0] - bound[1])
        edge_normal = np.cross(normal, edge_dir)
        edge_normal = edge_normal / np.linalg.norm(edge_normal) #TODO(ANDREW): Why is the edge_normal not used?
        pnt3 = bound[0] + edge_dir / 2

        return occ_utils.face_polygon([bound[1], bound[2], pnt3])