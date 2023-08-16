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

from OCC.Core.BRep import BRep_Tool
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend import TopologyUtils
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_FORWARD, TopAbs_REVERSED 
from OCC.Core.TopAbs import (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE,
                             TopAbs_SHELL, TopAbs_SOLID, TopAbs_COMPOUND,
                             TopAbs_COMPSOLID)
from OCC.Core.TopoDS import (
    TopoDS_Solid,
    TopoDS_Compound,
    TopoDS_CompSolid,
)
from OCC.Core.TopExp import topexp
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, 
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface, 
                              GeomAbs_BSplineSurface, GeomAbs_Line, GeomAbs_Circle, 
                              GeomAbs_Ellipse, GeomAbs_Hyperbola, GeomAbs_Parabola, 
                              GeomAbs_BezierCurve, GeomAbs_BSplineCurve, 
                              GeomAbs_OffsetCurve, GeomAbs_OtherCurve)
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties

# occwl
from occwl.edge_data_extractor import EdgeDataExtractor, EdgeConvexity
from occwl.edge import Edge
from occwl.face import Face
from occwl.solid import Solid
from occwl.uvgrid import uvgrid
from occwl.graph import face_adjacency



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


class AAGExtractor:
    def __init__(
        self, 
        step_file, 
        attribute_schema, 
        scale_body=True):
        self.step_file = step_file
        self.attribute_schema = attribute_schema
        self.scale_body = scale_body
        # whether to extract UV-grid
        self.use_uv = "UV-grid" in self.attribute_schema.keys()
        self.topchecker = TopologyChecker()
        if self.use_uv:
            # UV-gird size for face
            self.num_srf_u = self.attribute_schema["UV-grid"]["num_srf_u"]
            self.num_srf_v = self.attribute_schema["UV-grid"]["num_srf_v"]
            # UV-gird size for curve
            self.num_crv_u = self.attribute_schema["UV-grid"]["num_crv_u"]

    def process(self):
        """
        Creates a attributed adjacency graph from the given shape (Solid)

        Args:
            
        Returns:

        """
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

        # Build face adjacency graph with B-rep entities as node and edge features
        # occwl.Solid is a occwl warp for TopoDS_Shape
        try:
            graph = face_adjacency(Solid(self.body))
        except Exception as e:
            print(e)
            assert False, 'Wrong shape {}'.format(self.step_file)

        # get the attributes for faces
        graph_face_attr = []
        graph_face_grid = []
        # the FaceCentroidAttribute has xyz coordinate
        # so the length of face attributes should add 2 if containing centroid
        len_of_face_attr = len(self.attribute_schema["face_attributes"]) + \
            2 if "FaceCentroidAttribute" in self.attribute_schema["face_attributes"] else 0
        for face_idx in graph.nodes:
            # Get the B-rep face
            face = graph.nodes[face_idx]["face"]
            # get the attributes from face
            face_attr = self.extract_attributes_from_face(
                face.topods_shape())  # from occwl.Face to OCC.TopoDS_Face
            assert len_of_face_attr == len(face_attr)
            graph_face_attr.append(face_attr)
            # get the UV point grid from face
            if self.use_uv and self.num_srf_u and self.num_srf_v:
                uv_grid = self.extract_face_point_grid(face)
                assert uv_grid.shape[0] == 7
                graph_face_grid.append(uv_grid.tolist())

        graph_edge_attr = []
        graph_edge_grid = []
        for edge_idx in graph.edges:
            edge = graph.edges[edge_idx]["edge"]
            # Ignore dgenerate edges, e.g. at apex of cone
            if not edge.has_curve():
                continue
            # get the attributes from edge
            edge = edge.topods_shape() # from occwl.Edge to OCC.TopoDS_Edge
            edge_attr = self.extract_attributes_from_edge(edge)
            assert len(self.attribute_schema["edge_attributes"]) == len(edge_attr)
            graph_edge_attr.append(edge_attr)
            # get the UV point grid from edge
            if self.use_uv and self.num_crv_u:
                u_grid = self.extract_edge_point_grid(edge)
                assert u_grid.shape[0] == 12
                graph_edge_grid.append(u_grid.tolist())
        
        # get graph from nx.DiGraph
        edges = list(graph.edges)
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        graph = { 
            'edges': (src, dst),
            'num_nodes': len(graph.nodes)
        }

        return { 
            'graph': graph,
            'graph_face_attr': graph_face_attr,
            'graph_face_grid': graph_face_grid,
            'graph_edge_attr': graph_edge_attr,
            'graph_edge_grid': graph_edge_grid,
        }

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
    # Face Attributes Extractor
    ########################

    def extract_attributes_from_face(self, face) -> list:
        def plane_attribute(face):
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Plane:
                return 1.0
            return 0.0

        def cylinder_attribute(face):
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Cylinder:
                return 1.0
            return 0.0

        def cone_attribute(face):
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Cone:
                return 1.0
            return 0.0

        def sphere_attribute(face):
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Sphere:
                return 1.0
            return 0.0

        def torus_attribute(face):
            surf_type = BRepAdaptor_Surface(face).GetType()
            if surf_type == GeomAbs_Torus:
                return 1.0
            return 0.0

        def area_attribute(face):
            geometry_properties = GProp_GProps()
            brepgprop_SurfaceProperties(face, geometry_properties)
            return geometry_properties.Mass()

        def rational_nurbs_attribute(face):
            surf = BRepAdaptor_Surface(face)
            if surf.GetType() == GeomAbs_BSplineSurface:
                bspline = surf.BSpline()
            elif surf.GetType() == GeomAbs_BezierSurface:
                bspline = surf.Bezier()
            else:
                bspline = None
            
            if bspline is not None:
                if bspline.IsURational() or bspline.IsVRational():
                    return 1.0
            return 0.0

        def centroid_attribute(face):
            mass_props = GProp_GProps()
            brepgprop_SurfaceProperties(face, mass_props)
            gPt = mass_props.CentreOfMass()

            return gPt.Coord()

        face_attributes = []
        for attribute in self.attribute_schema["face_attributes"]:
            if attribute == "Plane":
                face_attributes.append(plane_attribute(face))
            elif attribute == "Cylinder":
                face_attributes.append(cylinder_attribute(face))
            elif attribute == "Cone":
                face_attributes.append(cone_attribute(face))
            elif attribute == "SphereFaceAttribute":
                face_attributes.append(sphere_attribute(face))
            elif attribute == "TorusFaceAttribute":
                face_attributes.append(torus_attribute(face))
            elif attribute == "FaceAreaAttribute":
                face_attributes.append(area_attribute(face))
            elif attribute == "RationalNurbsFaceAttribute":
                face_attributes.append(rational_nurbs_attribute(face))
            elif attribute == "FaceCentroidAttribute":
                face_attributes.extend(centroid_attribute(face))
            else:
                assert False, "Unknown face attribute"
        return face_attributes
        
    ########################
    # Edge attributes Extractor
    ########################

    def extract_attributes_from_edge(self, edge) -> list:
        def find_edge_convexity(edge, faces):
            edge_data = EdgeDataExtractor(Edge(edge), 
                faces, use_arclength_params=False)
            if not edge_data.good:
                # This is the case where the edge is a pole of a sphere
                return 0.0
            angle_tol_rads = 0.0872664626 # 5 degrees 
            convexity = edge_data.edge_convexity(angle_tol_rads)
            return convexity

        def convexity_attribute(convexity, attribute):
            if attribute == "Convex edge":
                return convexity == EdgeConvexity.CONVEX
            if attribute == "Concave edge":
                return convexity == EdgeConvexity.CONCAVE
            if attribute == "Smooth":
                return convexity == EdgeConvexity.SMOOTH
            assert False, "Unknown convexity"
            return 0.0

        def edge_length_attribute(edge):
            geometry_properties = GProp_GProps()
            brepgprop_LinearProperties(edge, geometry_properties)
            return geometry_properties.Mass()

        def circular_edge_attribute(edge):
            brep_adaptor_curve = BRepAdaptor_Curve(edge)
            curv_type = brep_adaptor_curve.GetType()
            if curv_type == GeomAbs_Circle:
                return 1.0
            return 0.0

        def closed_edge_attribute(edge):
            if BRep_Tool().IsClosed(edge):
                return 1.0
            return 0.0

        def elliptical_edge_attribute(edge):
            brep_adaptor_curve = BRepAdaptor_Curve(edge)
            curv_type = brep_adaptor_curve.GetType()
            if curv_type == GeomAbs_Ellipse:
                return 1.0
            return 0.0

        def helical_edge_attribute(edge):
            # We don't have this attribute in Open Cascade
            assert False, "Not implemented for the OCC pipeline"
            return 0.0

        def int_curve_edge_attribute(edge):
            # We don't have this attribute in Open Cascade
            assert False, "Not implemented for the OCC pipeline"
            return 0.0

        def straight_edge_attribute(edge):
            brep_adaptor_curve = BRepAdaptor_Curve(edge)
            curv_type = brep_adaptor_curve.GetType()
            if curv_type == GeomAbs_Line:
                return 1.0
            return 0.0

        def hyperbolic_edge_attribute(edge):
            if Edge(edge).curve_type() == "hyperbola":
                return 1.0
            return 0.0

        def parabolic_edge_attribute(edge):
            if Edge(edge).curve_type() == "parabola":
                return 1.0
            return 0.0

        def bezier_edge_attribute(edge):
            if Edge(edge).curve_type() == "bezier":
                return 1.0
            return 0.0

        def non_rational_bspline_edge_attribute(edge):
            occwl_edge = Edge(edge)
            if occwl_edge.curve_type() == "bspline" and not occwl_edge.rational():
                return 1.0
            return 0.0

        def rational_bspline_edge_attribute(edge):
            occwl_edge = Edge(edge)
            if occwl_edge.curve_type() == "bspline" and occwl_edge.rational():
                return 1.0
            return 0.0

        def offset_edge_attribute(edge):
            if Edge(edge).curve_type() == "offset":
                return 1.0
            return 0.0
        
        # get the faces from an edge
        top_exp = TopologyUtils.TopologyExplorer(self.body, ignore_orientation=True)
        faces_of_edge = [Face(f) for f in top_exp.faces_from_edge(edge)]

        attribute_list = self.attribute_schema["edge_attributes"]
        if "Concave edge" in attribute_list or \
            "Convex edge" in attribute_list or \
            "Smooth"  in attribute_list:
            convexity = find_edge_convexity(edge, faces_of_edge)
        edge_attributes = []
        for attribute in attribute_list:
            if attribute == "Concave edge":
                edge_attributes.append(convexity_attribute(convexity, attribute))
            elif attribute == "Convex edge":
                edge_attributes.append(convexity_attribute(convexity, attribute))
            elif attribute == "Smooth":
                edge_attributes.append(convexity_attribute(convexity, attribute))
            elif attribute == "EdgeLengthAttribute":
                edge_attributes.append(edge_length_attribute(edge))
            elif attribute == "CircularEdgeAttribute":
                edge_attributes.append(circular_edge_attribute(edge))
            elif attribute == "ClosedEdgeAttribute":
                edge_attributes.append(closed_edge_attribute(edge))
            elif attribute == "EllipticalEdgeAttribute":
                edge_attributes.append(elliptical_edge_attribute(edge))
            elif attribute == "HelicalEdgeAttribute":
                edge_attributes.append(helical_edge_attribute(edge))
            elif attribute == "IntcurveEdgeAttribute":
                edge_attributes.append(int_curve_edge_attribute(edge))
            elif attribute == "StraightEdgeAttribute":
                edge_attributes.append(straight_edge_attribute(edge))
            elif attribute == "HyperbolicEdgeAttribute":
                edge_attributes.append(hyperbolic_edge_attribute(edge))
            elif attribute == "ParabolicEdgeAttribute":
                edge_attributes.append(parabolic_edge_attribute(edge))
            elif attribute == "BezierEdgeAttribute":
                edge_attributes.append(bezier_edge_attribute(edge))
            elif attribute == "NonRationalBSplineEdgeAttribute":
                edge_attributes.append(non_rational_bspline_edge_attribute(edge))
            elif attribute == "RationalBSplineEdgeAttribute":
                edge_attributes.append(rational_bspline_edge_attribute(edge))
            elif attribute == "OffsetEdgeAttribute":
                edge_attributes.append(offset_edge_attribute(edge))
            else:
                assert False, "Unknown face attribute"
        return edge_attributes

    ########################
    # Face UV Point Grid Extractor
    ########################

    def extract_face_point_grid(self, face) -> np.array:
        """
        Extract a UV-Net point grid from the given face.

        Returns a tensor [ 7 x num_pts_u x num_pts_v ]

        For each point the values are 
        
            - x, y, z (point coords)
            - i, j, k (normal vector coordinates)
            - Trimming mast

        """
        points = uvgrid(face, self.num_srf_u, self.num_srf_v, method="point")
        normals = uvgrid(face, self.num_srf_u, self.num_srf_v, method="normal")
        mask = uvgrid(face, self.num_srf_u, self.num_srf_v, method="inside")

        # This has shape [ num_pts_u x num_pts_v x 7 ]
        single_grid = np.concatenate([points, normals, mask], axis=2)
       
        return np.transpose(single_grid, (2, 0, 1))

    ########################
    # Edge UV Point Grid Extractor
    ########################

    def extract_edge_point_grid(self, edge) -> np.array:
        """
        Extract a edge grid (aligned with the coedge direction).

        The edge grids will be of size

            [ 12 x num_u ]

        The values are

            - x, y, z    (coordinates of the points)
            - tx, ty, tz (tangent of the curve, oriented to match the coedge)
            - Lx, Ly, Lz (Normal for the left face)
            - Rx, Ry, Rz (Normal for the right face)
        """

        # get the faces from an edge
        top_exp = TopologyUtils.TopologyExplorer(self.body, ignore_orientation=True)
        faces_of_edge = [Face(f) for f in top_exp.faces_from_edge(edge)]

        edge_data = EdgeDataExtractor(Edge(edge), faces_of_edge, 
            num_samples=self.num_crv_u, use_arclength_params=True)
        if not edge_data.good:
            # We hit a problem evaluating the edge data.  This may happen if we have
            # an edge with not geometry (like the pole of a sphere).
            # In this case we return zeros
            return np.zeros((12, self.num_crv_u)) 

        single_grid = np.concatenate(
            [
                edge_data.points, 
                edge_data.tangents, 
                edge_data.left_normals,
                edge_data.right_normals
            ],
            axis = 1
        )
        return np.transpose(single_grid, (1, 0))


#######################################
# Calculate mean & std of attributes
#######################################

def check_zero_std(stat_data):
    std_face_attr = stat_data['std_face_attr']
    std_edge_attr = stat_data['std_edge_attr']
    if np.nonzero(std_face_attr)[0].shape[0] != len(std_face_attr) \
        or np.nonzero(std_edge_attr)[0].shape[0] != len(std_edge_attr):
        print('WARNING! has zero standard deviation.')


def find_standardization(data):
    '''
    Find mean and standard deviation of face and edge attributes
    Args:
    data (list): [filename, graph_data]
    '''
    all_face_attr = []
    all_edge_attr = []
    for one_sample in data:
        fn, graph = one_sample
        all_face_attr.extend(graph["graph_face_attr"])
        all_edge_attr.extend(graph["graph_edge_attr"])
    graph_face_attr = np.asarray(all_face_attr)
    graph_edge_attr = np.asarray(all_edge_attr)

    mean_face_attr = np.mean(graph_face_attr, axis=0)
    std_face_attr = np.std(graph_face_attr, axis=0)

    mean_edge_attr = np.mean(graph_edge_attr, axis=0)
    std_edge_attr = np.std(graph_edge_attr, axis=0)

    return {
            'mean_face_attr': mean_face_attr.tolist(),
            'std_face_attr': std_face_attr.tolist(),
            'mean_edge_attr': mean_edge_attr.tolist(),
            'std_edge_attr': std_edge_attr.tolist(),
        }

########################
# Save all step files in a json
########################

def load_json(pathname):
    with open(pathname, "r") as fp:
        return json.load(fp)


def save_json_data(pathname, data):
    """Export a data to a json file"""
    with open(pathname, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)


def initializer():
    import signal
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process_one_file(args):
    fn, feature_schema = args

    extractor = AAGExtractor(fn, feature_schema)
    out = extractor.process()

    return [str(fn.stem), out]


def main(args):
    step_path = Path(args.step_path)
    output_path = Path(args.output)
    if not output_path.exists():
        output_path.mkdir()

    feature_list_path = None
    if args.feature_list is not None:
        feature_list_path = Path(args.feature_list)
    
    parent_folder = Path(__file__).parent.parent
    if feature_list_path is None:
        feature_list_path = parent_folder / "feature_lists/all.json"
    feature_schema = load_json(feature_list_path)
    step_files = list(step_path.glob("*.st*p"))

    pool = Pool(processes=args.num_workers, initializer=initializer)
    try:
        results = list(tqdm(
            pool.imap(
                process_one_file, zip(step_files, repeat(feature_schema))), 
            total=len(step_files)))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()

    save_json_data(
        osp.join(output_path, 'graphs.json'), results)
    
    attr_stat = find_standardization(results)
    check_zero_std(attr_stat)
    save_json_data(
        osp.join(output_path, 'attr_stat.json'), attr_stat)
    
    gc.collect()
    print(f"Processed {len(results)} files.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_path", type=str, required=True, help="Path to load the step files from")
    parser.add_argument("--output", type=str, required=True, help="Path to the save intermediate brep data")
    parser.add_argument("--feature_list", type=str, required=False, help="Optional path to the feature lists")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers")
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