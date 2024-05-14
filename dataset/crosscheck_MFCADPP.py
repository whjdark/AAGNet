# -*- coding: utf-8 -*-
import pathlib
import argparse
import json
import os
from tqdm import tqdm

from OCC.Core.TopoDS import (
    TopoDS_Solid,
    TopoDS_Compound,
    TopoDS_CompSolid,
)
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Extend import TopologyUtils
from OCC.Extend.TopologyUtils import TopologyExplorer

# occwl
from occwl.solid import Solid
from occwl.graph import face_adjacency



def list_face(shape):
    '''
    input
        shape: TopoDS_Shape
    output
        fset: {TopoDS_Face}
    '''
    """
    fset = set()
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        s = exp.Current()
        exp.Next()
        face = topods.Face(s)
        fset.add(face)
    return list(fset)
    """
    topo = TopologyExplorer(shape)

    return list(topo.faces())


class TopologyChecker():
    # modified from BREPNET: https://github.com/AutodeskAILab/BRepNet/blob/master/pipeline/extract_brepnet_data_from_step.py
    def __init__(self):
        pass

    def find_edges_from_wires(self, top_exp):
        edge_set = set()
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for edge in wire_exp.ordered_edges():
                edge_set.add(edge)
        return edge_set

    def find_edges_from_top_exp(self, top_exp):
        edge_set = set(top_exp.edges())
        return edge_set

    def check_closed(self, body):
        # In Open Cascade, unlinked (open) edges can be identified
        # as they appear in the edges iterator when ignore_orientation=False
        # but are not present in any wire
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        edges_from_wires = self.find_edges_from_wires(top_exp)
        edges_from_top_exp = self.find_edges_from_top_exp(top_exp)
        missing_edges = edges_from_top_exp - edges_from_wires
        return len(missing_edges) == 0

    def check_manifold(self, top_exp):
        faces = set()
        for shell in top_exp.shells():
            for face in top_exp._loop_topo(TopAbs_FACE, shell):
                if face in faces:
                    return False
                faces.add(face)
        return True

    def check_unique_coedges(self, top_exp):
        coedge_set = set()
        for loop in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(loop)
            for coedge in wire_exp.ordered_edges():
                orientation = coedge.Orientation()
                tup = (coedge, orientation)
                # We want to detect the case where the coedges
                # are not unique
                if tup in coedge_set:
                    return False
                coedge_set.add(tup)
        return True

    def __call__(self, body):
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)
        if top_exp.number_of_faces() == 0:
            print('Empty shape') 
            return False
        # OCC.BRepCheck, perform topology and geometricals check
        analyzer = BRepCheck_Analyzer(body)
        if not analyzer.IsValid(body):
            print('BRepCheck_Analyzer found defects') 
            return False
        # other topology check
        if not self.check_manifold(top_exp):
            print("Non-manifold bodies are not supported")
            return False
        if not self.check_closed(body):
            print("Bodies which are not closed are not supported")
            return False
        if not self.check_unique_coedges(top_exp):
            print("Bodies where the same coedge is uses in multiple loops are not supported")
            return False
        return True


def load_body_from_step(step_file):
    """
    Load the body from the step file.  
    We expect only one body in each file
    """
    assert pathlib.Path(step_file).suffix in ['.step', '.stp']
    reader = STEPControl_Reader()
    reader.ReadFile(str(step_file))
    reader.TransferRoots()
    shape = reader.OneShape()
    return shape


def load_json(pathname):
    with open(pathname, "r") as fp:
        return json.load(fp)


def get_filenames(path, suffix):
    path = pathlib.Path(path)
    files = list(
        x for x in path.rglob(suffix)
    )
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to load the dataset from")
    args = parser.parse_args()
    # load dataset
    aag_path = os.path.join(args.dataset, 'aag', 'graphs.json')
    step_path = os.path.join(args.dataset, 'steps')
    labels_path = os.path.join(args.dataset, 'labels')
    # AGG exists
    agg_exist = os.path.exists(aag_path)
    if agg_exist:
        print('AGG json exists')
        try:
            agg = load_json(aag_path)
        except Exception as e:
            assert False, e
    step_files = get_filenames(step_path, f"*.st*p")
    labels_files = get_filenames(labels_path, '*.json')
    # check number of files
    if agg_exist:
        assert len(agg) == len(step_files), \
        'number of AGG ({}) is not equal to number of step files ({})' .format(
            len(agg), len(step_files))
    assert len(step_files) == len(labels_files), \
        'number of label files ({}) is not equal to number of step files ({})' .format(
            len(labels_files), len(step_files))
    
    wrong_files = []
    if agg_exist:
        # loop over AAgraph, step_file, label 
        for agg_data, step_file, labels_file in tqdm(
            zip(agg, step_files, labels_files), total=len(step_files)):
            # check file name
            fn, graph = agg_data
            assert fn == step_file.stem
            assert step_file.stem == labels_file.stem
            # load step, label json
            try:
                shape = load_body_from_step(step_file)
            except Exception as e:
                print(fn, e)
                wrong_files.append((step_file, labels_file))
                continue
            try: 
                label_data = load_json(labels_file)
            except Exception as e:
                print(fn, e)
                wrong_files.append((step_file, labels_file))
                continue
            # crosscehck labels
            faces_list = list_face(shape)
            num_faces = len(faces_list)
            # check length of aag equals to number of faces
            if num_faces != graph['graph']['num_nodes']:
                print('File {} have wrong number of labels {} with AAG faces {}. '.format(
                    fn, num_faces, graph['graph']['num_nodes']))
                wrong_files.append((step_file, labels_file))
                continue
            # check length of label
            seg_label = label_data
            # check map between face id and segmentaion label
            if num_faces != len(seg_label):
                print('File {} have wrong number of seg labels {} with step faces {}. '.format(
                    fn, len(seg_label), num_faces))
                wrong_files.append((step_file, labels_file))
                continue
    else:
        # loop over step_file, label
        topochecker = TopologyChecker()
        for step_file, labels_file in tqdm(
            zip(step_files, labels_files), total=len(step_files)):
            # check file name
            fn = step_file.stem
            assert step_file.stem == labels_file.stem
            # load step, label json
            try:
                shape = load_body_from_step(step_file)
            except Exception as e:
                print('file', fn)
                print(e)
                wrong_files.append((step_file, labels_file))
                continue
            try: 
                label_data = load_json(labels_file)
            except Exception as e:
                print('file', fn)
                print(e)
                wrong_files.append((step_file, labels_file))
                continue
            # check shape is TopoDS_Solid
            if not isinstance(shape, TopoDS_Solid):
                print('{} is {}, not supported'.format(fn, type(shape)))
                wrong_files.append((step_file, labels_file))
                continue
            # check shape topology
            if not topochecker(shape):
                print("{} has wrong topology".format(fn))
                wrong_files.append((step_file, labels_file))
                continue
            # check generated shape can be exported to face_adjacency
            try:
                graph = face_adjacency(Solid(shape))
            except Exception as e:
                print('Wrong {} with Exception: {}'.format(fn)) 
                print(e)
                wrong_files.append((step_file, labels_file))
                continue
            # crosscehck labels
            faces_list = list_face(shape)
            num_faces = len(faces_list)
            # check length of label
            seg_label = label_data
            # check map between face id and segmentaion label
            if num_faces != len(seg_label):
                print('File {} have wrong number of seg labels {} with step faces {}. '.format(
                    fn, len(seg_label), num_faces))
                wrong_files.append((step_file, labels_file))
                continue
    # delete wrong steps and labels or not
    if len(wrong_files):
        print('delete following wrong files:')
        print(wrong_files)
        inputs = input('[Y/N]: ')
        if (inputs == 'Y') or (inputs == 'y'):
            for wrong_file in wrong_files:
                step_f ,label_f = wrong_file
                os.remove(step_f)
                os.remove(label_f)
                print(step_f ,label_f, 'deleted')
            if agg_exist:
                os.remove(aag_path)
                print(aag_path, 'deleted, please regenerat AAG')

    print('Finished')