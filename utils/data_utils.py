import random
import os.path as osp
import json
import pathlib

import numpy as np
import torch
import dgl
from scipy.spatial.transform import Rotation

from OCC.Core.STEPControl import STEPControl_Reader



def bounding_box_pointcloud(pts: torch.Tensor):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
    return torch.tensor(box)


def bounding_box_uvgrid(inp: torch.Tensor):
    pts = inp[..., :3].reshape((-1, 3))
    mask = inp[..., 6].reshape(-1)
    point_indices_inside_faces = mask == 1
    pts = pts[point_indices_inside_faces, :]
    return bounding_box_pointcloud(pts)


def center_and_scale_uvgrid(inp: torch.Tensor, return_center_scale=False):
    inp = inp.transpose(1, 3) # channel last
    bbox = bounding_box_uvgrid(inp)
    diag = bbox[1] - bbox[0]
    scale = 2.0 / max(diag[0], diag[1], diag[2])
    center = 0.5 * (bbox[0] + bbox[1])
    inp[..., :3] -= center
    inp[..., :3] *= scale
    inp = inp.transpose(1, 3) # channel first
    if return_center_scale:
        return inp, center, scale
    return 


def center_and_scale(data: torch.Tensor):
    data["graph"].ndata["grid"], center, scale = center_and_scale_uvgrid(
        data["graph"].ndata["grid"], return_center_scale=True
    )
    if "grid" in data["graph"].edata.keys():
        egrid = data["graph"].edata["grid"]
        egrid = egrid.transpose(1, 2) # channel last
        egrid[..., :3] -= center
        egrid[..., :3] *= scale
        egrid = egrid.transpose(1, 2) # channel first
        data["graph"].edata["grid"] = egrid
    return data


def standardization(data, stat):
    data["graph"].ndata["x"] -= stat['mean_face_attr']
    data["graph"].ndata["x"] /= stat['std_face_attr']
    data["graph"].edata["x"] -= stat['mean_edge_attr']
    data["graph"].edata["x"] /= stat['std_edge_attr']
    return data


def get_random_rotation():
    """
    Get a random rotation in 90 degree increments along the canonical axes
    """
    axes = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
    ]
    angles = [0.0, 90.0, 180.0, 270.0]
    axis = random.choice(axes)
    angle_radians = np.radians(random.choice(angles))
    return Rotation.from_rotvec(angle_radians * axis)


def rotate_uvgrid(inp, rotation):
    """
    Rotate the node features in the graph by a given rotation
    """
    inp = inp.transpose(1, 3) # channel last
    Rmat = torch.tensor(rotation.as_matrix()).float()
    orig_size = inp[..., :3].size()
    inp[..., :3] = torch.mm(inp[..., :3].reshape(-1, 3), Rmat).reshape(
        orig_size
    )  # Points
    inp[..., 3:6] = torch.mm(inp[..., 3:6].reshape(-1, 3), Rmat).reshape(
        orig_size
    )  # Normals/tangents
    inp = inp.transpose(1, 3) # channel first
    return inp


def load_body_from_step(step_file):
    """
    Load the body from the step file.  
    We expect only one body in each file
    """
    assert pathlib.Path(step_file).suffix in ['.step', '.stp', '.STEP', '.STP']
    reader = STEPControl_Reader()
    reader.ReadFile(str(step_file))
    reader.TransferRoots()
    shape = reader.OneShape()
    return shape


def load_json_or_pkl(pathname):
    # try to load dataset from pickel first
    pkl_path = str(pathname).split('.')[0] + '.pkl'
    if osp.exists(pkl_path):
        return torch.load(pkl_path)
    else: # if no pkl exists, load from json
        with open(pathname, "r") as fp:
            return json.load(fp)


def load_one_graph(fn, data):
    # Create the graph using the edges and number of nodes
    edges = tuple(data['graph']['edges'])
    num_nodes = data['graph']['num_nodes']
    dgl_graph = dgl.graph(edges, num_nodes=num_nodes)

    # Convert node attributes to PyTorch tensors and add them to the graph
    node_attributes = data['graph_face_attr']
    node_attributes = np.array(node_attributes)
    node_attributes = torch.from_numpy(node_attributes).type(torch.float32)
    dgl_graph.ndata["x"] = node_attributes

    # Convert and add node grid attributes if they are present
    node_grid_attributes = data['graph_face_grid']
    if len(node_grid_attributes) > 0:
        node_grid_attributes = np.array(node_grid_attributes)
        node_grid_attributes = torch.from_numpy(node_grid_attributes).type(torch.float32)
        dgl_graph.ndata["grid"] = node_grid_attributes

    # Convert edge attributes to PyTorch tensors and add them to the graph
    edge_attributes = data['graph_edge_attr']
    edge_attributes = np.array(edge_attributes)
    edge_attributes = torch.from_numpy(edge_attributes).type(torch.float32)
    dgl_graph.edata["x"] = edge_attributes

    # Convert and add edge grid attributes if they are present
    edge_grid_attributes = data['graph_edge_grid']
    if len(edge_grid_attributes) > 0:
        edge_grid_attributes = np.array(edge_grid_attributes)
        edge_grid_attributes = torch.from_numpy(edge_grid_attributes).type(torch.float32)
        dgl_graph.edata["grid"] = edge_grid_attributes
    
    sample = {"graph": dgl_graph, "filename": fn}
    return sample


def load_statistics(stat_path):
    stat = load_json_or_pkl(stat_path)
    mean_face_attr = np.array(stat['mean_face_attr'])
    std_face_attr = np.array(stat['std_face_attr'])
    mean_edge_attr = np.array(stat['mean_edge_attr'])
    std_edge_attr = np.array(stat['std_edge_attr'])
    stat['mean_face_attr'] = torch.from_numpy(mean_face_attr)
    stat['std_face_attr'] = torch.from_numpy(std_face_attr)
    stat['mean_edge_attr'] = torch.from_numpy(mean_edge_attr)
    stat['std_edge_attr'] = torch.from_numpy(std_edge_attr)
    # if the std is 0, we set the std to 1
    eps = 1e-8
    stat['std_face_attr'][stat['std_face_attr'] < eps] = 1.
    stat['std_edge_attr'][stat['std_edge_attr'] < eps] = 1.
    return stat