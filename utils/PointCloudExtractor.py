# -*- coding: utf-8 -*-
import argparse
from multiprocessing.pool import Pool
import gc
import json
import os.path as osp
import numpy as np
from pathlib import Path
from tqdm import tqdm
from itertools import repeat

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend import TopologyUtils
from OCC.Core.TopAbs import (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE,
                             TopAbs_SHELL, TopAbs_SOLID, TopAbs_COMPOUND,
                             TopAbs_COMPSOLID)
from OCC.Core.TopoDS import (
    TopoDS_Solid,
    TopoDS_Compound,
    TopoDS_CompSolid,
)

from OCC.Extend.TopologyUtils import TopologyExplorer

# occwl
from occwl.solid import Solid
from occwl.face import Face
from occwl.uvgrid import uvgrid



def scale_solid_to_unit_box(solid):
    if isinstance(solid, Solid):
        return solid.scale_to_unit_box(copy=True)
    solid = Solid(solid, allow_compound=True)
    solid = solid.scale_to_unit_box(copy=True)
    return solid.topods_shape()


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


class PointCloudExtractor:
    def __init__(
        self, 
        step_file, 
        scale_body=True):
        self.step_file = step_file
        self.num_points = 6
        self.scale_body = scale_body
        self.topchecker = TopologyChecker()
        
    def process(self):
        """
        create a point cloud from a STEP file

        Args:
            
        Returns:

        """
        # import time # run on workers=1
        # start = time.time()
        # Load the body from the STEP file
        self.body = self.load_body_from_step()
        assert self.body is not None, \
            "the shape {} is non-manifold or open".format(self.step_file)
        assert self.topchecker(self.body), \
            "the shape {} has wrong topology".format(self.step_file)
        assert isinstance(self.body, TopoDS_Solid), \
            'file {} is {}, not TopoDS_Solid'.format(self.step_file, type(self.body))
        
        # We want to apply a transform so that the solid
        # is centered on the origin and scaled so it just fits
        # into a box [-1, 1]^3
        if self.scale_body:
            self.body = scale_solid_to_unit_box(self.body)

        topo = TopologyExplorer(self.body)
        faces = list(topo.faces())
        # sample point from each faces
        pcs_in_faces = []
        for face in faces:
            face = Face(face)
            pcs_in_face = self.extract_face_point_grid(face)
            # drop 4 points from 6*6 to 32
            pcs_in_faces.append(pcs_in_face[0:32, :])
        # if the num of faces is less than 64, skip this file
        # else pad zero to 64
        if len(pcs_in_faces) > 128:
            print("the shape {} has more than 128 faces".format(self.step_file))
            return None
        
        if len(pcs_in_faces) < 128:
            for i in range(128 - len(pcs_in_faces)):
                pcs_in_faces.append(np.zeros((32, 6), dtype=np.float32))
        
        pcs_in_faces = np.array(pcs_in_faces, dtype=np.float32)
        # add new axis
        pcs_in_faces = np.expand_dims(pcs_in_faces, axis=0)

        # print('duration time: ', time.time() - start)

        return pcs_in_faces

    ########################
    # Step Loader
    ########################

    def load_body_from_step(self):
        """
        Load the body from the step file.  
        We expect only one body in each file
        """
        step_filename_str = str(self.step_file)
        reader = STEPControl_Reader()
        reader.ReadFile(step_filename_str)
        reader.TransferRoots()
        shape = reader.OneShape()
        return shape

    ########################
    # Face UV Point Grid Extractor
    ########################

    def extract_face_point_grid(self, face) -> np.array:
        """
        Extract a UV-Net point grid from the given face.

        Returns a tensor [ num_pts_u x num_pts_v, 6 ]

        For each point the values are 
        
            - x, y, z (point coords)
            - i, j, k (normal vector coordinates)

        """
        points = uvgrid(face, self.num_points, self.num_points, method="point")
        normals = uvgrid(face, self.num_points, self.num_points, method="normal")

        # This has shape [ num_pts_u x num_pts_v x 6 ]
        single_grid = np.concatenate([points, normals], axis=2)

        return single_grid.reshape(self.num_points * self.num_points, 6)


########################
# Save all step files in a json
########################


def initializer():
    import signal
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process_one_file(args):
    fn, output_path = args

    extractor = PointCloudExtractor(fn)
    out = extractor.process()
    
    # save as numpy array
    if out is not None:
        np.save(osp.join(output_path, str(fn.stem)+'.npy'), out)

    return 


def main(args):
    step_path = Path(args.step_path)
    output_path = Path(args.output)
    if not output_path.exists():
        output_path.mkdir()

    step_files = list(step_path.glob("*.st*p"))

    pool = Pool(processes=args.num_workers, initializer=initializer)
    try:
        results = list(tqdm(
            pool.imap(
                process_one_file, 
                zip(step_files, repeat(output_path))), 
            total=len(step_files)))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    
    gc.collect()
    print(f"Processed {len(results)} files.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_path", type=str, required=True, help="Path to load the step files from")
    parser.add_argument("--output", type=str, required=True, help="Path to the save intermediate brep data")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker threads")
    args = parser.parse_args()

    main(args)

    # pathname = "E:\\learning_from_Brep\\feature_lists\\all.json"
    # brep_path = "E:\\autodesk_BRep_code\\BRepNet\\fusion360\\breps\\step\\50203_ae7e8919_0.stp"
    # with open(pathname, encoding='utf8') as data_file:
    #     feature_scheme = json.load(data_file)
    # extractor = AAGExtractor(brep_path, feature_scheme)
    # out = extractor.process()
    # print(len(graph_data))

    # print(graph_face_attr.shape)
    # print(graph_edge_attr.shape)

    # print(graph_face_grid.shape)
    # print(graph_edge_grid.shape)

    # print(graph_face_grid[0:20,6,:,:])
    
    # predict_points_OnFace = None
    # predict_namesofpart = None
    # predict_FacesId = None

    # for d in filenames_predict:
    #    cur_points_OnFace, NamesofPart, FacesID = load_h5_name_id(os.path.join(path, d), 1)
    #    if predict_points_OnFace is None:
    #        predict_points_OnFace = cur_points_OnFace
    #        predict_namesofpart = NamesofPart
    #        predict_FacesId = FacesID
    #    else:
    #        predict_points_OnFace = np.vstack((predict_points_OnFace, cur_points_OnFace))
    #        predict_namesofpart = np.hstack((predict_namesofpart, NamesofPart))
    #        predict_FacesId = np.hstack((predict_FacesId, FacesID), dtype=int)
    # predict_namesofpart = predict_namesofpart.reshape((len(predict_namesofpart), 1))
    # print(predict_points_OnFace.shape, predict_namesofpart.shape, predict_FacesId.shape)

    # cur_points_OnFace, NamesofPart, FacesID = load_h5_name_id(os.path.join(path, d), 1)
    # print(cur_points_OnFace.shape)
    # data = np.load("E:\\AAGNet\\dataset\\debug_data\\pcs\\20221121_154647_0.npy")
    # print(data.shape)
    # predict_points_OnFace = data
    # predict_namesofpart = np.array(['01'])
    # predict_namesofpart = predict_namesofpart.reshape((len(predict_namesofpart), 1))
    # predict_FacesId = np.array([i for i in range(predict_points_OnFace.shape[1])]).reshape(1, 64)
    # print(predict_FacesId.shape)