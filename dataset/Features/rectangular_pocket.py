import Utils.occ_utils as occ_utils

from Features.machining_features import MachiningFeature


class RectangularPocket(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = 4
        self.bound_type = 4
        self.depth_type = "blind"
        self.feat_type = "rectangular_pocket"

    def _add_sketch(self, bound):
        return occ_utils.face_polygon(bound[:4])