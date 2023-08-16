# -*- coding: utf-8 -*-
"""Creates dataset from random combination of machining features

Used to generate dataset of stock cube with machining features applied to them.
The number of machining features is defined by the combination range.
To change the parameters of each machining feature, please see parameters.py
"""

from itertools import combinations_with_replacement
from itertools import repeat
import Utils.shape as shape
import random
import os
import gc
import pickle
import time
from tqdm import tqdm
from multiprocessing.pool import Pool

from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.STEPConstruct import stepconstruct_FindEntity
from OCC.Core.TCollection import TCollection_HAsciiString
from OCC.Core.TopoDS import (
    TopoDS_Solid,
    TopoDS_Compound,
    TopoDS_CompSolid,
)
from OCC.Extend.DataExchange import STEPControl_Writer

import Utils.occ_utils as occ_utils
import feature_creation



def shape_with_fid_to_step(filename, shape, id_map, save_face_label=False):
    """Save shape to a STEP file format.

    :param filename: Name to save shape as.
    :param shape: Shape to be saved.
    :param id_map: Variable mapping labels to faces in shape.
    :return: None
    """
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)

    if save_face_label:
        finderp = writer.WS().TransferWriter().FinderProcess()
        faces = occ_utils.list_face(shape)
        loc = TopLoc_Location()
        for face in faces:
            item = stepconstruct_FindEntity(finderp, face, loc)
            if item is None:
                print(face)
                continue
            item.SetName(TCollection_HAsciiString(str(id_map[face])))

    writer.Write(filename)


def directive(combo, count):
    shape_name = str(count)
    shapes, labels = feature_creation.shape_from_directive(combo)

    seg_map, inst_map = labels[0], labels[1]
    faces_list = occ_utils.list_face(shape)
    # Create map between face id and segmentaion label
    seg_label = feature_creation.get_segmentaion_label(faces_list, seg_map)
    # Create relation_matrix describing the feature instances
    relation_matrix = feature_creation.get_instance_label(faces_list, len(seg_map), inst_map)

    return shapes, shape_name, (seg_label, relation_matrix)


def save_shape(shape, step_path, label_map):
    print(f"Saving: {step_path}")
    shape_with_fid_to_step(step_path, shape, label_map)


def save_label(shape_name, pathname, seg_label, relation_matrix, bottom_label):
    import json
    """
    Export a data to a json file
    """
    data = [
        [shape_name, {'seg': seg_label, 'inst': relation_matrix, 'bottom': bottom_label}]
    ]
    with open(pathname, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)


def generate_shape(args):
    """
    Generate num_shapes random shapes in dataset_dir
    :param arg: List of [shape directory path, shape name, machining feature combo]
    :return: None
    """
    dataset_dir, combo = args
    f_name, combination = combo

    num_try = 0 # first try
    while True:
        num_try += 1
        print('try count', num_try)
        if num_try > 3:
            # fails too much, pass
            print('number of fails > 3, pass')
            break

        try:
            shape, labels = feature_creation.shape_from_directive(combination)
        except Exception as e:
            print('Fail to generate:')
            print(e)
            continue

        if shape is None:
            print('generated shape is None')
            continue
        
        # from topologyCheker import TopologyChecker
        # check shape topology
        # too slow, we perform TopologyChecker after step generated
        # topochecker = TopologyChecker()
        # if not topochecker(shape):
        #     print("generated shape has wrong topology")
        #     continue
    
        # check generated shape has supported type (TopoDS_Solid, TopoDS_Compound, TopoDS_CompSolid)
        if not isinstance(shape, (TopoDS_Solid, TopoDS_Compound, TopoDS_CompSolid)):
            print('generated shape is {}, not supported'.format(type(shape)))
            continue
        
        # get the corresponding semantic segmentaion, instance and bottom labels
        seg_map, inst_label, bottom_map = labels
        faces_list = occ_utils.list_face(shape)
        if len(faces_list) == 0:
            print('empty shape')
            continue
        # Create map between face id and segmentaion label
        seg_label = feature_creation.get_segmentaion_label(faces_list, seg_map)
        if len(seg_label) != len(faces_list):
            print('generated shape has wrong number of seg labels {} with step faces {}. '.format(
                len(seg_label), len(faces_list)))
            continue
        # Create relation_matrix describing the feature instances
        relation_matrix = feature_creation.get_instance_label(faces_list, len(seg_map), inst_label)
        if len(relation_matrix) != len(faces_list):
            print('generated shape has wrong number of instance labels {} with step faces {}. '.format(
                len(relation_matrix), len(faces_list)))
            continue
        # Create map between face id and botto identification label
        bottom_label = feature_creation.get_segmentaion_label(faces_list, bottom_map)
        if len(bottom_label) != len(faces_list):
            print('generated shape haswrong number of bottom labels {} with step faces {}. '.format(
                len(bottom_label), len(faces_list)))
            continue
        # save step and its labels
        shape_name = str(f_name)
        step_path = os.path.join(dataset_dir, 'steps')
        label_path = os.path.join(dataset_dir, 'labels')
        step_path = os.path.join(step_path, shape_name + '.step')
        label_path = os.path.join(label_path, shape_name + '.json')
        try:
            save_shape(shape, step_path, seg_map)
            save_label(shape_name, label_path, seg_label, relation_matrix, bottom_label)
        except Exception as e:
            print('Fail to save:')
            print(e)
            continue
        print('SUCCESS')
        break # success
    return


def initializer():
    import signal
    """
    Ignore CTRL+C in the worker process.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)



if __name__ == '__main__':
    # Parameters to be set before use
    dataset_scale = 'large' # or large
    num_features = 24 # for large dataset, use full classes
    # for tiny dataset, only common features
    # Through hole, Blind hole, Rectangular pocket, Rectangular through slot, Round, Chamfer
    tiny_dataset_cand_feats = [1, 12, 14, 6, 0, 23]
    cand_feat_weights = [0.6, 0.6, 1, 1, 0.3, 0.3]
    dataset_dir = 'data'
    combo_range = [5, 14]
    num_samples = 22000
    num_workers = 12

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    step_path = os.path.join(dataset_dir, 'steps')
    label_path = os.path.join(dataset_dir, 'labels')
    if not os.path.exists(step_path):
        os.mkdir(step_path)
    if not os.path.exists(label_path):
        os.mkdir(label_path)

    # old feature combination generation
    # combos = []
    # for num_combo in range(combo_range[0], combo_range[1]):
    #     combos += list(combinations_with_replacement(range(num_features), num_combo))

    # print('total combinations: ', len(combos))
    # random.shuffle(combos)
    # test_combos = combos[:num_samples]
    # del combos

    combos = []
    for idx in range(num_samples):
        num_inter_feat = random.randint(combo_range[0], combo_range[1])
        if dataset_scale == 'large':
            combo = [random.randint(0, num_features-1) for _ in range(num_inter_feat)] # no stock face
        elif dataset_scale == 'tiny':
            combo = random.choices(tiny_dataset_cand_feats, weights=cand_feat_weights, k=num_inter_feat)

        now =  time.localtime()
        now_time = time.strftime("%Y%m%d_%H%M%S", now)
        file_name = now_time + '_' + str(idx)
        combos.append((file_name, combo))

    if num_workers == 1:
        for combo in combos:
            generate_shape((dataset_dir, combo))
    elif num_workers > 1: # multiprocessing
        pool = Pool(processes=num_workers, initializer=initializer)
        try:
            result = list(tqdm(pool.imap(generate_shape, zip(repeat(dataset_dir), combos)), 
                            total=len(combos)))
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
    else:
        AssertionError('error number of workers')
    
    gc.collect()


