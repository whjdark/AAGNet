import Utils.occ_utils as occ_utils
from Features.machining_features import MachiningFeature


class RectangularThroughSlot(MachiningFeature):
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        super().__init__(shape, label_map, min_len, clearance, feat_names)
        self.shifter_type = 1
        self.bound_type = 1
        self.depth_type = "through"
        self.feat_type = "rectangular_through_slot"

    def _add_sketch(self, bound):
        return occ_utils.face_polygon(bound[:4])
