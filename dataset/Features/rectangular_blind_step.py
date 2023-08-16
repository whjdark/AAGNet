import Utils.occ_utils as occ_utils

from Features.machining_features import MachiningFeature


class RectangularBlindStep(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = 2
        self.bound_type = 2
        self.depth_type = "blind"
        self.feat_type = "rectangular_blind_step"

    def _add_sketch(self, bound):
        return occ_utils.face_polygon(bound[:4])