import random

import numpy as np

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeEdge
from OCC.Core.gp import gp_Pnt
from OCC.Extend.TopologyUtils import TopologyExplorer

import Utils.shape_factory as shape_factory
import Utils.parameters as param
import Utils.occ_utils as occ_utils

from Features.o_ring import ORing
from Features.through_hole import ThroughHole
from Features.round import Round
from Features.chamfer import Chamfer
from Features.triangular_passage import TriangularPassage
from Features.rectangular_passage import RectangularPassage
from Features.six_sides_passage import SixSidesPassage
from Features.triangular_through_slot import TriangularThroughSlot
from Features.rectangular_through_slot import RectangularThroughSlot
from Features.circular_through_slot import CircularThroughSlot
from Features.rectangular_through_step import RectangularThroughStep
from Features.two_sides_through_step import TwoSidesThroughStep
from Features.slanted_through_step import SlantedThroughStep
from Features.blind_hole import BlindHole
from Features.triangular_pocket import TriangularPocket
from Features.rectangular_pocket import RectangularPocket
from Features.six_sides_pocket import SixSidesPocket
from Features.circular_end_pocket import CircularEndPocket
from Features.rectangular_blind_slot import RectangularBlindSlot
from Features.v_circular_end_blind_slot import VCircularEndBlindSlot
from Features.h_circular_end_blind_slot import HCircularEndBlindSlot
from Features.triangular_blind_step import TriangularBlindStep
from Features.circular_blind_step import CircularBlindStep
from Features.rectangular_blind_step import RectangularBlindStep



feat_names = ['chamfer', 'through_hole', 'triangular_passage', 'rectangular_passage', '6sides_passage',
              'triangular_through_slot', 'rectangular_through_slot', 'circular_through_slot',
              'rectangular_through_step', '2sides_through_step', 'slanted_through_step', 'Oring', 'blind_hole',
              'triangular_pocket', 'rectangular_pocket', '6sides_pocket', 'circular_end_pocket',
              'rectangular_blind_slot', 'v_circular_end_blind_slot', 'h_circular_end_blind_slot',
              'triangular_blind_step', 'circular_blind_step', 'rectangular_blind_step', 'round', 'stock']

feat_classes = {"chamfer": Chamfer, "through_hole": ThroughHole, "triangular_passage": TriangularPassage,
                "rectangular_passage": RectangularPassage, "6sides_passage": SixSidesPassage,
                "triangular_through_slot": TriangularThroughSlot, "rectangular_through_slot": RectangularThroughSlot,
                "circular_through_slot": CircularThroughSlot, "rectangular_through_step": RectangularThroughStep,
                "2sides_through_step": TwoSidesThroughStep, "slanted_through_step": SlantedThroughStep, "Oring": ORing,
                "blind_hole": BlindHole, "triangular_pocket": TriangularPocket, "rectangular_pocket": RectangularPocket,
                "6sides_pocket": SixSidesPocket, "circular_end_pocket": CircularEndPocket,
                "rectangular_blind_slot": RectangularBlindSlot, "v_circular_end_blind_slot": VCircularEndBlindSlot,
                "h_circular_end_blind_slot": HCircularEndBlindSlot, "triangular_blind_step": TriangularBlindStep,
                "circular_blind_step": CircularBlindStep, "rectangular_blind_step": RectangularBlindStep,
                "round": Round}

through_blind_features = ["triangular_passage", "rectangular_passage", "6sides_passage", "triangular_pocket",
                          "rectangular_pocket", "6sides_pocket", "through_hole", "blind_hole", "circular_end_pocket",
                          "Oring"]


def triangulate_shape(shape):
    linear_deflection = 0.1
    angular_deflection = 0.5
    mesh = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    mesh.Perform()
    assert mesh.IsDone()


def generate_stock_dims(larger_stock):
    if larger_stock: # too much features need larger stock for avoiding wrong topology
        stock_min_x = param.stock_min_x * 2
        stock_min_y = param.stock_min_y * 2
        stock_min_z = param.stock_min_z * 2
    else:
        stock_min_x = param.stock_min_x
        stock_min_y = param.stock_min_y
        stock_min_z = param.stock_min_z
    param.stock_dim_x = random.uniform(stock_min_x, param.stock_max_x)
    param.stock_dim_y = random.uniform(stock_min_y, param.stock_max_y)
    param.stock_dim_z = random.uniform(stock_min_z, param.stock_max_z)


def rearrange_combo(combination):
    transition_feats = []
    step_feats = []
    slot_feats = []
    through_feats = []
    blind_feats = []
    o_ring_feats = []

    for cnt, val in enumerate(combination):
        if val == param.feat_names.index("chamfer") or val == param.feat_names.index("round"):
            transition_feats.append(val)
        elif val == param.feat_names.index("rectangular_through_step") \
                or val == param.feat_names.index("2sides_through_step") \
                or val == param.feat_names.index("slanted_through_step") \
                or val == param.feat_names.index("triangular_blind_step") \
                or val == param.feat_names.index("circular_blind_step") \
                or val == param.feat_names.index("rectangular_blind_step"):
            step_feats.append(val)

        elif val == param.feat_names.index("triangular_through_slot") \
                or val == param.feat_names.index("rectangular_through_slot") \
                or val == param.feat_names.index("circular_through_slot") \
                or val == param.feat_names.index("rectangular_blind_slot") \
                or val == param.feat_names.index("v_circular_end_blind_slot") \
                or val == param.feat_names.index("h_circular_end_blind_slot"):
            slot_feats.append(val)

        elif val == param.feat_names.index("through_hole") \
                or val == param.feat_names.index("triangular_passage") \
                or val == param.feat_names.index("rectangular_passage") \
                or val == param.feat_names.index("6sides_passage"):
            through_feats.append(val)

        elif val == param.feat_names.index("blind_hole") \
                or val == param.feat_names.index("triangular_pocket") \
                or val == param.feat_names.index("rectangular_pocket") \
                or val == param.feat_names.index("6sides_pocket") \
                or val == param.feat_names.index("circular_end_pocket"):
            blind_feats.append(val)

        elif val == param.feat_names.index("Oring"):
            o_ring_feats.append(val)

    new_combination = step_feats + slot_feats + through_feats + blind_feats + o_ring_feats + transition_feats

    return new_combination


def rearrange_combo_planar(combination):
    transition_feats = []
    step_feats = []
    slot_feats = []
    through_feats = []
    blind_feats = []

    for cnt, val in enumerate(combination):
        if val == param.feat_names.index("chamfer"):
            transition_feats.append(val)
        elif val == param.feat_names.index("rectangular_through_step") \
                or val == param.feat_names.index("2sides_through_step") \
                or val == param.feat_names.index("slanted_through_step") \
                or val == param.feat_names.index("triangular_blind_step") \
                or val == param.feat_names.index("rectangular_blind_step"):
            step_feats.append(val)

        elif val == param.feat_names.index("triangular_through_slot") \
                or val == param.feat_names.index("rectangular_through_slot") \
                or val == param.feat_names.index("rectangular_blind_slot"):
            slot_feats.append(val)

        elif val == param.feat_names.index("triangular_passage") \
                or val == param.feat_names.index("rectangular_passage") \
                or val == param.feat_names.index("6sides_passage"):
            through_feats.append(val)
        elif val == param.feat_names.index("triangular_pocket") \
                or val == param.feat_names.index("rectangular_pocket") \
                or val == param.feat_names.index("6sides_pocket"):
            blind_feats.append(val)

    new_combination = step_feats + slot_feats + through_feats + blind_feats + transition_feats

    return new_combination


def shape_from_directive(combo):
    try_cnt = 0
    find_edges = True
    combo = rearrange_combo(combo) # rearrange machining feature combinations
    count = 0
    bounds = []

    while True:
        # random stock size
        if len(combo) >= 10:
            # too much features need larger stock for avoiding wrong topology
            generate_stock_dims(larger_stock=True)
        else:
            generate_stock_dims(larger_stock=False)
        # create stock
        shape = BRepPrimAPI_MakeBox(param.stock_dim_x, param.stock_dim_y, param.stock_dim_z).Shape()
        # non-feature faces are labeled as stok
        label_map = shape_factory.map_from_name(shape, param.feat_names.index('stock'))

        for fid in combo:
            feat_name = param.feat_names[fid]
            if feat_name == "chamfer":
                edges = occ_utils.list_edge(shape)
                # create new feature object
                new_feat = feat_classes[feat_name](shape, label_map, param.min_len,
                                                   param.clearance, param.feat_names, edges)
                shape, label_map, edges = new_feat.add_feature()

                if len(edges) == 0:
                    break

            elif feat_name == "round":
                if find_edges:
                    edges = occ_utils.list_edge(shape)
                    find_edges = False

                new_feat = feat_classes[feat_name](shape, label_map, param.min_len,
                                                   param.clearance, param.feat_names, edges)
                shape, label_map, edges = new_feat.add_feature()

                if len(edges) == 0:
                    break

            else:
                triangulate_shape(shape) # mesh curved surface ???
                new_feat = feat_classes[feat_name](shape, label_map, param.min_len, param.clearance, param.feat_names)
                if count == 0:
                    shape, label_map, bounds = new_feat.add_feature(bounds, find_bounds=True)

                    if feat_name in through_blind_features:
                        count += 1

                else: # I think it should find bounds after each feature created besides from inner bounds
                    # may slow generation speed
                    shape, label_map, bounds = new_feat.add_feature(bounds, find_bounds=True) # orignial: False
                    count += 1

        if shape is not None:
            break

        try_cnt += 1
        if try_cnt > len(combo):
            shape = None
            label_map = None
            break
            
    return shape, label_map


def display_bounds(bounds, display, color):
    for bound in bounds:
        rect = [gp_Pnt(bound[0][0], bound[0][1], bound[0][2]),
                gp_Pnt(bound[1][0], bound[1][1], bound[1][2]),
                gp_Pnt(bound[2][0], bound[2][1], bound[2][2]),
                gp_Pnt(bound[3][0], bound[3][1], bound[3][2]),
                gp_Pnt(bound[0][0], bound[0][1], bound[0][2])]

        wire_sect = BRepBuilderAPI_MakeWire()

        for i in range(len(rect) - 1):
            edge_sect = BRepBuilderAPI_MakeEdge(rect[i], rect[i+1]).Edge()
            wire_sect.Add(edge_sect)

        sect = wire_sect.Wire()

        display.DisplayShape(sect, update=True, color=color)

    return display


def get_segmentaion_label(faces_list, seg_map):
    '''
    Create map between face id and segmentaion label
    '''
    faceid_label_map = {}
    for face in faces_list:
        face_idx = faces_list.index(face)    
        faceid_label_map[face_idx] = seg_map[face]

    return faceid_label_map


def get_instance_label(faces_list, num_faces, inst_map):
    '''
    Create relation_matrix describing the feature instances
    '''
    relation_matrix = np.zeros((num_faces, num_faces), dtype=np.uint8)

    for inst in inst_map:
        for row_inst_face in inst:
            if row_inst_face not in faces_list:
                print('WARNING! mssing face', row_inst_face.__hash__()) 
                continue
            row_face_idx = faces_list.index(row_inst_face) 
            # In the face_idx row，all instance faces are labeled as 1
            for col_inst_face in inst:
                if col_inst_face not in faces_list:
                    print('WARNING! mssing face', col_inst_face.__hash__()) 
                    continue
                col_face_idx = faces_list.index(col_inst_face)
                # pythonocc index starts from 1
                relation_matrix[row_face_idx][col_face_idx] = 1

    assert relation_matrix.nonzero(), 'relation_matrix is empty'
    assert np.allclose(relation_matrix, relation_matrix.T), 'relation_matrix is not symmetric'

    return relation_matrix.tolist()


def save_json_data(pathname, data):
    import json
    """Export a data to a json file"""
    with open(pathname, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)


if __name__ == '__main__':
    combo = [6, 6, 6, 6, 6, 6, 6, 6]

    shape, label_map = shape_from_directive(combo)

    print(label_map)
    seg_map, inst_label, bottom_map = label_map
    # for f in seg_map.keys():
    #     print(f.__hash__())
    # print('--------------------------------------------------')
    # for f in bottom_map.keys():
    #     print(f.__hash__())

    faces_list = occ_utils.list_face(shape)
    # Create map between face id and segmentaion label
    seg_label = get_segmentaion_label(faces_list, seg_map)
    assert len(seg_label) == len(faces_list)
    print(seg_label)

    # Create relation_matrix describing the feature instances
    relation_matrix = get_instance_label(faces_list, len(faces_list), inst_label)
    assert len(seg_label) == len(relation_matrix)
    for row in relation_matrix:
        print(row)

    # Create map between face id and botto identification label
    bottom_label = get_segmentaion_label(faces_list, bottom_map)
    assert len(seg_label) == len(bottom_label)
    print(bottom_label)

    # save step
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs

    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    writer.Write('test.step')

    data = [
        ['test', {'seg': seg_label, 'inst': relation_matrix, 'bottom': bottom_label}]
    ]
    # save label
    save_json_data('test.json', data)
