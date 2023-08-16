import random
import math
import numpy as np
import numba as nbv

from OCC.Core.BRepFeat import BRepFeat_MakePrism
from OCC.Core.gp import gp_Dir
from OCC.Core.TopoDS import TopoDS_Face

import Utils.occ_utils as occ_utils
import Utils.geom_utils as geom_utils
import Utils.geom_utils_numba as geom_utils_nb
import Utils.shape_factory as shape_factory
import Utils.parameters as param
import Utils.numba_vec as nbv

import OCCUtils.edge
import OCCUtils.face

from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopExp import topexp
from OCC.Extend.TopologyUtils import TopologyExplorer

from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Ax1
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.TopoDS import topods_Face
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface


class MachiningFeature:
    def __init__(self, shape, label_map, min_len, clearance, feat_names):
        self.shape = shape
        self.min_len = min_len
        self.clearance = clearance
        self.bounds = []
        self.shifter_type = None
        self.bound_type = None
        self.points = []
        self.depth_type = None
        self.labels = label_map
        self.feat_names = feat_names
        self.feat_type = None

    def _get_bounds(self):
        if self.bound_type == 1:
            self._bound_1()
        elif self.bound_type == 2:
            self._bound_2()
        elif self.bound_type == 3:
            self._bound_3()
        elif self.bound_type == 4:
            self._bound_inner()
        else:
            print(f"Bound type of {self.bound_type} does not exist.")

    def _get_depth(self, bound, triangles):
        """Selects appropiate method for finding depth of feature.

        :param bound:
        :param triangles:
        :return:
        """
        if self.depth_type == "through":
            return self._depth_through()
        elif self.depth_type == "blind":
            return self._depth_blind(bound, triangles)
        else:
            print(f"Depth type of {self.depth_type} does not exist.")

    def _depth_blind(self, bound, triangles):
        """Selects depth of blind feature.

        Find bounds of blind depth and randomly selects the depth.

        :param bound:
        :param triangles:
        :return: depth of blind machining feature
        """
        thres = self.min_len + self.clearance
        depths = []
        self.points = geom_utils.points_inside_rect(bound[0], bound[1], bound[2], bound[3], 0.2)

        for pnt in self.points:
            triangles = np.array(triangles, dtype=np.float64)
            dpt = geom_utils_nb.ray_triangle_set_intersect(pnt, bound[4], triangles)
            if dpt < 0.0:
                continue
            if dpt + 1e-6 < thres:
                return np.NINF

            depths.append(dpt)

        depths.sort()

        d_min = min(depths)

        if d_min < 0:
            return np.NINF

        return random.uniform(self.min_len, d_min - self.clearance)

    def _depth_through(self):
        depth = max([param.stock_dim_x, param.stock_dim_y, param.stock_dim_z])
        return depth

    def _shifter2(self, max_bound):
        dir_w = max_bound[2] - max_bound[1]
        dir_h = max_bound[0] - max_bound[1]
        old_w = np.linalg.norm(dir_w)
        old_h = np.linalg.norm(dir_h)
        dir_w = dir_w / old_w
        dir_h = dir_h / old_h

        scale_w = random.uniform(0.1, 1.0)
        scale_h = random.uniform(0.1, 1.0)
        new_w = max(param.min_len, old_w * scale_w)
        new_h = max(param.min_len, old_h * scale_h)

        if self.shifter_type == 1:
            offset_w = random.uniform(0.0, old_w - new_w)
            max_bound[1] = max_bound[1] + offset_w * dir_w

        if self.shifter_type == 4:
            offset_w = random.uniform(0.0, old_w - new_w)
            offset_h = random.uniform(0.0, old_h - new_h)
            max_bound[1] = max_bound[1] + offset_w * dir_w + offset_h * dir_h

        max_bound[0] = max_bound[1] + new_h * dir_h

        if not self.shifter_type == 3:
            max_bound[2] = max_bound[1] + new_w * dir_w
            max_bound[3] = max_bound[0] + new_w * dir_w
        else:
            max_bound[3] = max_bound[2] + new_h * dir_h

        return max_bound

    def _shifter(self, bounds_max):
        dir_w = nbv.sub(bounds_max[1], bounds_max[2])
        dir_h = nbv.sub(bounds_max[1], bounds_max[0])
        old_w = nbv.calc_l2_norm(dir_w)
        old_h = nbv.calc_l2_norm(dir_h)
        dir_w = nbv.div(old_w, dir_w)
        dir_h = nbv.div(old_h, dir_h)

        scale_w = random.uniform(0.1, 1.0)
        scale_h = random.uniform(0.1, 1.0)
        new_w = max(param.min_len, old_w * scale_w)
        new_h = max(param.min_len, old_h * scale_h)

        if self.shifter_type == 1:
            offset_w = random.uniform(0.0, old_w - new_w)
            new_dir_w = nbv.mul(offset_w, dir_w)
            bounds_max[1] = nbv.add(bounds_max[1], new_dir_w)

        if self.shifter_type == 4:
            offset_w = random.uniform(0.0, old_w - new_w)
            offset_h = random.uniform(0.0, old_h - new_h)
            new_dir_w = nbv.mul(offset_w, dir_w)
            new_dir_h = nbv.mul(offset_h, dir_h)
            bounds_max[1] = nbv.add(nbv.add(bounds_max[1], new_dir_w), new_dir_h)

        new_dir_h = nbv.mul(new_h, dir_h)
        bounds_max[0] = nbv.add(bounds_max[1], new_dir_h)

        if not self.shifter_type == 3:
            new_dir_w = nbv.mul(new_w, dir_w)
            bounds_max[2] = nbv.add(bounds_max[1], new_dir_w)
            bounds_max[3] = nbv.add(bounds_max[0], new_dir_w)
        else:
            new_dir_h = nbv.mul(new_h, dir_h)
            bounds_max[3] = nbv.add(bounds_max[2], new_dir_h)

        return bounds_max

    def _triangulation_from_face(self, face):
        aLoc = TopLoc_Location()
        aTriangulation = BRep_Tool().Triangulation(face, aLoc)
        aTrsf = aLoc.Transformation()

        aNodes = aTriangulation.Nodes()
        aTriangles = aTriangulation.Triangles()

        pts = []
        for i in range(1, aTriangulation.NbNodes() + 1):
            pt = aNodes.Value(i)
            pt.Transform(aTrsf)
            pts.append([pt.X(), pt.Y(), pt.Z()])

        triangles = []
        vt_map = {}
        et_map = {}
        for i in range(1, aTriangulation.NbTriangles() + 1):
            n1, n2, n3 = aTriangles.Value(i).Get()
            pids = [n1 - 1, n2 - 1, n3 - 1]
            pids.sort()
            triangles.append((pids[0], pids[1], pids[2]))

            for pid in pids:
                if pid in vt_map:
                    vt_map[pid].append(i - 1)
                else:
                    vt_map[pid] = [i - 1]

            edges = [(pids[0], pids[1]), (pids[0], pids[2]), (pids[1], pids[2])]
            for edge in edges:
                if edge in et_map:
                    et_map[edge].append(i - 1)
                else:
                    et_map[edge] = [i - 1]

        return pts, triangles, vt_map, et_map

    def _triangles_from_faces(self, faces):
        tri_list = []
        for face in faces:
            pts, triangles, vt_map, et_map = self._triangulation_from_face(face)
            for tri in triangles:
                tri_list.append((pts[tri[0]], pts[tri[1]], pts[tri[2]]))

        return tri_list

    def _rect_size(self, rect):
        dir_w = nbv.sub(rect[1], rect[2])
        dir_h = nbv.sub(rect[1], rect[0])
        width = nbv.calc_l2_norm(dir_w)
        height = nbv.calc_l2_norm(dir_h)

        return width, height

    def _face_filter(self, shape, num_edges):
        result = []
        faces = occ_utils.list_face(shape)

        for face in faces:
            surf = BRepAdaptor_Surface(face, True)
            surf_type = surf.GetType()

            if surf_type != GeomAbs_Plane:
                continue

            if num_edges == 0:
                result.append(face)
                continue

            face_util = OCCUtils.face.Face(face)
            normal = occ_utils.as_list(occ_utils.normal_to_face_center(face))
            for wire in face_util.topo.wires():
                edges = [edge for edge in OCCUtils.face.WireExplorer(wire).ordered_edges()]
                if len(edges) < 4:
                    continue

                good_edge = []
                for edge in edges:
                    if occ_utils.type_edge(edge) != 'line':
                        good_edge.append(False)
                        continue

                    else:
                        good_edge.append(True)

                    face_adjacent = occ_utils.face_adjacent(shape, face, edge)
                    assert face_adjacent is not None

                for i in range(len(edges)):
                    j = (i + 1) % len(edges)
                    k = (i + 2) % len(edges)

                    if not good_edge[i]:
                        continue

                    if num_edges == 1:
                        result.append([face, edges[i]])
                        continue

                    if not good_edge[j]:
                        continue

                    pnt = np.array(occ_utils.as_list(topexp.FirstVertex(edges[i], True)))
                    pntj = np.array(occ_utils.as_list(topexp.FirstVertex(edges[j], True)))
                    pntk = np.array(occ_utils.as_list(topexp.FirstVertex(edges[k], True)))
                    normal = np.array(normal, dtype=np.float64)
                    if not geom_utils_nb.point_in_polygon(pnt, np.array([pntj, pntk]), normal=normal):
                        continue

                    if num_edges == 2:
                        result.append([face, edges[i], edges[j]])
                        continue

                    if not good_edge[k]:
                        continue

                    pnt = np.array(occ_utils.as_list(topexp.LastVertex(edges[k], True)))
                    if not geom_utils_nb.point_in_polygon(pnt, np.array([pntj, pntk]), normal=normal):
                        continue

                    if num_edges == 3:
                        result.append([face, edges[i], edges[j], edges[k]])

        return result

    def _sample_points_inside_face(self, face):
        tri_pnts, triangles, _, et_map = self._triangulation_from_face(face)
        sample_points = []

        # sketch on 3 edges
        for edge in et_map:
            if len(et_map[edge]) > 1:
                pt1 = np.asarray(tri_pnts[edge[0]])
                pt2 = np.asarray(tri_pnts[edge[1]])
                e_dir = nbv.sub(pt2, pt1)
                e_len = nbv.calc_l2_norm(e_dir)

                if e_len < self.min_len + 2 * self.clearance:
                    continue

                sample_points.append((0.5 * pt1 + 0.5 * pt2).tolist())

        for tri in triangles:
            pt1 = np.asarray(tri_pnts[tri[0]])
            pt2 = np.asarray(tri_pnts[tri[1]])
            pt3 = np.asarray(tri_pnts[tri[2]])
            if geom_utils_nb.outer_radius_triangle(pt1, pt2, pt3) < self.min_len + 2 * self.clearance:
                continue

            sample_points.append(((pt1 + pt2 + pt3) / 3).tolist())
            sample_points.append((0.2 * pt1 + 0.4 * pt2 + 0.4 * pt3).tolist())
            sample_points.append((0.4 * pt1 + 0.2 * pt2 + 0.4 * pt3).tolist())
            sample_points.append((0.4 * pt1 + 0.4 * pt2 + 0.2 * pt3).tolist())

        return sample_points

    def _bound_inner(self):
        fe_list = self._face_filter(self.shape, num_edges=0)

        for face in fe_list:
            normal = np.array(occ_utils.as_list(occ_utils.normal_to_face_center(face)))
            sample_pnts = np.array(self._sample_points_inside_face(face))

            edges = occ_utils.list_edge(face)

            for apnt in sample_pnts:
                # Finds apnt closest to edge of B-Rep face
                dist, pnt = occ_utils.dist_point_to_edges(apnt, edges)
                pnt = np.array(pnt)

                if dist >= param.min_len / 2 + param.clearance:
                    dir_w = nbv.sub(pnt, apnt)
                    len_w = nbv.norm(dir_w)
                    unit_dir_w = nbv.div(len_w, dir_w)
                    dir_h = nbv.cross(unit_dir_w, normal)
                    dist -= param.clearance

                    part_a = nbv.sub(nbv.mul(dist, unit_dir_w), apnt)
                    part_b = nbv.add(nbv.mul(dist, unit_dir_w), apnt)
                    part_c = nbv.mul(dist, dir_h)

                    pnt0 = nbv.sub(part_c, part_a)
                    pnt1 = nbv.add(part_c, part_a)
                    pnt2 = nbv.add(part_c, part_b)
                    pnt3 = nbv.sub(part_c, part_b)

                    bound = np.array((pnt0, pnt1, pnt2, pnt3, -normal))

                    intersect = self._possible_to_machine(bound, normal, fe_list)
                    if not intersect:
                        self.bounds.append(bound)

    def _bound_1(self):
        concave_edges = self._find_concave_edges(self.shape)
        fe_list = self._face_filter(self.shape, num_edges=1)
        faces = occ_utils.list_face(self.shape)

        for item in fe_list:
            face = item[0]
            edge = item[1]

            # Only allow feature creation on convex edges
            if edge in concave_edges:
                continue

            pts, triangles, vt_map, et_map = self._triangulation_from_face(face)
            segs = []
            for et in et_map:
                # Ensure the facet edge is on the context edge of the BRep face
                if len(et_map[et]) == 1:
                    segs.append([pts[et[0]], pts[et[1]]])
            pts = np.asarray(pts)
            normal = np.array(occ_utils.as_list(occ_utils.normal_to_face_center(face)))

            pnt1 = np.array(occ_utils.as_list(topexp.FirstVertex(edge, True)))
            pnt2 = np.array(occ_utils.as_list(topexp.LastVertex(edge, True)))

            edge_dir = nbv.sub(pnt1, pnt2)
            edge_len = nbv.calc_l2_norm(edge_dir)

            if edge_len < (2 * param.clearance) + param.min_len:
                continue

            # Adjust point coordinates and edge length for clearance
            edge_unit_dir = nbv.div(edge_len, edge_dir)
            pnt1 = nbv.add(pnt1, nbv.mul(param.clearance, edge_unit_dir))
            pnt2 = nbv.add(pnt2, nbv.mul(param.clearance, -edge_unit_dir))
            edge_len -= 2 * param.clearance

            edge_normal = nbv.cross(normal, edge_unit_dir)
            edge_normal = nbv.div(nbv.calc_l2_norm(edge_normal), edge_normal)

            # Create sample points along the context edge
            num_sample = int(edge_len / param.min_len)
            sample_pnts = [pnt1 + t * param.min_len * edge_unit_dir for t in range(num_sample)]
            sample_pnts.append(pnt2)
            sample_pnts = np.array(sample_pnts, dtype=np.float64)
            segs = np.array(segs, dtype=np.float64)

            inter_pnts = []

            for i in range(sample_pnts.shape[0]):
                intersects = geom_utils_nb.ray_segment_set_intersect(sample_pnts[i], edge_normal, segs)
                intersects.sort()
                # The first intersection point is pnt itself
                for inter in intersects:
                    if inter > 0:
                        inter_pnts.append(sample_pnts[i] + inter * edge_normal)
                        break

            assert len(inter_pnts) == len(sample_pnts)
            pts = np.append(pts, inter_pnts, axis=0)

            inter_pnts = np.array(inter_pnts, dtype=np.float64)

            for i in range(sample_pnts.shape[0] - 1):
                vec1 = nbv.sub(sample_pnts[i], inter_pnts[i])

                for j in range(i + 1, sample_pnts.shape[0]):
                    vec2 = nbv.sub(sample_pnts[j], inter_pnts[j])
                    verts = np.array([sample_pnts[i] + vec1, sample_pnts[i], sample_pnts[j], sample_pnts[j] + vec2])
                    bound = geom_utils_nb.search_rect_inside_bound_1(verts, vec1, vec2, pts)
                    if bound is not None:
                        # Ensure that there is clearance from the opposite edge to the context edge
                        w, h = self._rect_size(bound)
                        if h >= param.min_len + param.clearance:
                            bound = self._shrink_bound_1(bound)
                            bound = np.append(bound, [-normal], axis=0)

                            intersect = self._possible_to_machine(bound, normal, faces)
                            if not intersect:
                                self.bounds.append(bound)

    def _shrink_bound_1(self, bound):
        dir_h_1 = nbv.sub(bound[1], bound[0])
        len_h_1 = nbv.norm(dir_h_1)
        dir_h_2 = nbv.sub(bound[2], bound[3])
        len_h_2 = nbv.norm(dir_h_2)

        if len_h_1 <= len_h_2:
            dir_h = nbv.mul(param.clearance / len_h_1, dir_h_1)
            h_dif = dir_h_1 - dir_h_2
            bound[0] = nbv.sub(dir_h, bound[0])
            bound[3] = nbv.sub(dir_h, bound[3]) + h_dif
        else:
            dir_h = nbv.mul(param.clearance / len_h_2, dir_h_2)
            h_dif = dir_h_2 - dir_h_1
            bound[0] = nbv.sub(dir_h, bound[0]) + h_dif
            bound[3] = nbv.sub(dir_h, bound[3])

        return bound

    def _shrink_bound_2(self, bound):
        dir_w = nbv.sub(bound[1], bound[2])
        dir_h = nbv.sub(bound[1], bound[0])
        old_w = nbv.norm(dir_w)
        old_h = nbv.norm(dir_h)
        dir_w = nbv.mul(param.clearance / old_w, dir_w)
        dir_h = nbv.mul(param.clearance / old_h, dir_h)

        bound[0] = nbv.sub(dir_h, bound[0])
        bound[2] = nbv.sub(dir_w, bound[2])
        bound[3] = nbv.sub(dir_w, nbv.sub(dir_h, bound[3]))

        return bound

    def _shrink_bound_3(self, bound):
        dir_h_1 = nbv.sub(bound[1], bound[0])
        len_h_1 = nbv.norm(dir_h_1)
        dir_h_2 = nbv.sub(bound[2], bound[3])
        len_h_2 = nbv.norm(dir_h_2)

        if len_h_1 <= len_h_2:
            dir_h = nbv.mul(param.clearance / len_h_1, dir_h_1)
            h_dif = dir_h_1 - dir_h_2
            bound[0] = nbv.sub(dir_h, bound[0])
            bound[3] = nbv.sub(dir_h, bound[3]) + h_dif
        else:
            dir_h = nbv.mul(param.clearance / len_h_2, dir_h_2)
            h_dif = dir_h_2 - dir_h_1
            bound[0] = nbv.sub(dir_h, bound[0]) + h_dif
            bound[3] = nbv.sub(dir_h, bound[3])

        return bound

    def _bound_2(self):
        concave_edges = self._find_concave_edges(self.shape)

        fe_list = self._face_filter(self.shape, num_edges=2)
        faces = occ_utils.list_face(self.shape)

        for item in fe_list:
            face = item[0]
            edge1 = item[1]
            edge2 = item[2]

            if edge1 in concave_edges or edge2 in concave_edges:
                continue

            pnt0 = np.array(occ_utils.as_list(topexp.FirstVertex(edge1, True)))
            pnt1 = np.array(occ_utils.as_list(topexp.FirstVertex(edge2, True)))
            pnt2 = np.array(occ_utils.as_list(topexp.LastVertex(edge2, True)))

            edge_angle = self._angle_between_edges(pnt0, pnt1, pnt2)

            if edge_angle != 90.0:
                continue

            pts, triangles, vt_map, et_map = self._triangulation_from_face(face)
            pts = np.asarray(pts)

            vec0 = nbv.sub(pnt1, pnt0)
            vec2 = nbv.sub(pnt1, pnt2)

            verts = np.array([pnt1 + vec0, pnt1, pnt1 + vec2, pnt1 + vec0 + vec2])
            bound = geom_utils_nb.search_rect_inside_bound_2(verts, vec0, vec2, pts)

            if bound is not None:
                w, h = self._rect_size(bound)
                if w >= param.min_len + param.clearance and h >= param.min_len + param.clearance:
                    normal = np.array(occ_utils.as_list(occ_utils.normal_to_face_center(face)))
                    bound = self._shrink_bound_2(bound)
                    bound = np.append(bound, [-normal], axis=0)

                    intersect = self._possible_to_machine(bound, normal, faces)
                    if not intersect:
                        self.bounds.append(bound)

    def _bound_3(self):
        concave_edges = self._find_concave_edges(self.shape)
        fe_list = self._face_filter(self.shape, num_edges=3)
        faces = occ_utils.list_face(self.shape)

        for item in fe_list:
            face = item[0]
            edge1 = item[1]
            edge2 = item[2]
            edge3 = item[3]

            if edge1 in concave_edges or edge2 in concave_edges or edge3 in concave_edges:
                continue

            pts, triangles, vt_map, et_map = self._triangulation_from_face(face)
            pts = np.asarray(pts)

            v0 = np.array(occ_utils.as_list(topexp.FirstVertex(edge1, True)))
            v1 = np.array(occ_utils.as_list(topexp.FirstVertex(edge2, True)))
            v2 = np.array(occ_utils.as_list(topexp.FirstVertex(edge3, True)))
            v3 = np.array(occ_utils.as_list(topexp.LastVertex(edge3, True)))

            vertices = np.array([v0, v1, v2, v3], dtype=np.float64)
            bound = geom_utils_nb.search_rect_inside_bound_3(vertices, pts)

            if bound is not None:
                w, h = self._rect_size(bound)

                if w >= param.min_len and h >= param.min_len + param.clearance:
                    normal = np.array(occ_utils.as_list(occ_utils.normal_to_face_center(face)))
                    bound = self._shrink_bound_3(bound)
                    bound = np.append(bound, [-normal], axis=0)
                    intersect = self._possible_to_machine(bound, normal, faces)
                    if not intersect:
                        self.bounds.append(bound)

    def _possible_to_machine(self, bound, normal, faces):
        intersect = False

        centroid_x = (bound[0][0] + bound[1][0] + bound[2][0] + bound[3][0]) / 4
        centroid_y = (bound[0][1] + bound[1][1] + bound[2][1] + bound[3][1]) / 4
        centroid_z = (bound[0][2] + bound[1][2] + bound[2][2] + bound[3][2]) / 4
        centroid = (centroid_x, centroid_y, centroid_z)

        points = np.array((bound[0], bound[1], bound[2], bound[3], centroid))

        for face in faces:
            tri_list = []
            pts, triangles, vt_map, et_map = self._triangulation_from_face(face)

            for tri in triangles:
                tri_list.append((pts[tri[0]], pts[tri[1]], pts[tri[2]]))

            for pnt in points:
                tri_array = np.array(tri_list, dtype=np.float64)
                dpt = geom_utils_nb.ray_triangle_set_intersect(pnt, normal, tri_array)

                if dpt != np.NINF:
                    intersect = True
                    continue

            if intersect:
                continue

        return intersect

    def _angle_between_edges(self, pnt0, pnt1, pnt2):
        vec_a = pnt1 - pnt0
        vec_b = pnt2 - pnt1

        vec_dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        angle_cos = vec_dot / norm_a / norm_b
        angle_rad = np.arccos(angle_cos)
        angle = angle_rad * 180 / math.pi

        return angle

    def _find_concave_edges(self, shape):
        concave = []

        topo = TopologyExplorer(shape)
        for edge in topo.edges():
            faces = list(topo.faces_from_edge(edge))
            if len(faces) == 1:
                continue

            s = edge_dihedral(edge, faces)

            if s == -1:
                concave.append(edge)

        return concave

    def _add_sketch(self, bound):
        return None

    def _apply_feature(self, old_shape, old_labels, feat_type, feat_face, depth_dir, bound_max):
        feature_maker = BRepFeat_MakePrism()
        feature_maker.Init(old_shape, feat_face, TopoDS_Face(), occ_utils.as_occ(depth_dir, gp_Dir), False, False)
        feature_maker.Build()

        feature_maker.Perform(np.linalg.norm(depth_dir))
        shape = feature_maker.Shape()
        # find map map between modified faces on old shape and new generated faces
        fmap = shape_factory.map_face_before_and_after_feat(old_shape, feature_maker)
        # bottom face is parallel to the depth direction
        # special case bottom face is normal to the depth direction
        if self.feat_type == 'rectangular_through_slot' or \
            self.feat_type == 'rectangular_through_step' or \
            self.feat_type == 'rectangular_blind_step': # we also consider 'rectangular_blind_step' feature as special case
            # find the normal vector of the depth direction, which is the direction of feature height
            dir_h = nbv.sub(bound_max[1], bound_max[0])
            feat_dir = occ_utils.as_occ(dir_h, gp_Dir)
        else:
            feat_dir = occ_utils.as_occ(depth_dir, gp_Dir)
        new_labels = shape_factory.map_from_shape_and_name(fmap, old_labels, shape, self.feat_names.index(feat_type), feat_dir)
        
        return shape, new_labels

    def add_feature(self, bounds, find_bounds=True):
        """Adds machining feature to current shape.

        :param bounds:
        :param find_bounds:
        :return:
        """
        try:
            if find_bounds is True:
                self._get_bounds()
            else:
                self.bounds = bounds

            if len(self.bounds) < 1:
                return self.shape, self.labels, self.bounds

            feat_face = None
            faces = occ_utils.list_face(self.shape)
            triangles = self._triangles_from_faces(faces)

            random.shuffle(self.bounds)
            depth = np.NINF

            try_cnt = 0
            while try_cnt < len(self.bounds):
                bound_max = random.choice(self.bounds)
                bound_max = self._shifter(bound_max)

                depth = self._get_depth(bound_max, triangles)

                if depth <= 0:
                    try_cnt += 1
                    continue

                feat_face = self._add_sketch(bound_max)
                try_cnt = len(self.bounds)

        except Exception as e:
            print(e)
            return self.shape, self.labels, bounds

        if feat_face is None:
            return self.shape, self.labels, bounds
        
        feat_dir = bound_max[4]
        shape, labels = self._apply_feature(self.shape, self.labels, self.feat_type, 
                                            feat_face, feat_dir * depth, bound_max)
        
        topo = TopologyExplorer(shape)

        if topo.number_of_solids() > 1:
            return self.shape, self.labels, bounds

        return shape, labels, self.bounds


def ask_point_uv2(xyz, face):
    """
    This is a general function which gives the uv coordinates from the xyz coordinates.
    The uv value is not normalised.
    """
    gpPnt = gp_Pnt(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    surface = BRep_Tool().Surface(face)

    sas = ShapeAnalysis_Surface(surface)
    gpPnt2D = sas.ValueOfUV(gpPnt, 0.01)
    uv = list(gpPnt2D.Coord())

    return uv


def ask_face_centroid(face):
    """
    Get centroid of B-Rep face.
    """
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop
    mass_props = GProp_GProps()
    brepgprop.SurfaceProperties(face, mass_props)
    gPt = mass_props.CentreOfMass()

    return gPt.Coord()


def ask_point_normal_face(uv, face):
    """
    Ask the normal vector of a point given the uv coordinate of the point on a face
    """
    face_ds = topods_Face(face)
    surface = BRep_Tool().Surface(face_ds)
    props = GeomLProp_SLProps(surface, uv[0], uv[1], 1, 1e-6)

    gpDir = props.Normal()
    if face.Orientation() == TopAbs_REVERSED:
        gpDir.Reverse()

    return gpDir.Coord()


def ask_edge_midpnt_tangent(edge):
    """
    Ask the midpoint of an edge and the tangent at the midpoint
    """
    result = BRep_Tool.Curve(edge)  # result[0] is the handle of curve;result[1] is the umin; result[2] is umax
    tmid = (result[1] + result[2]) / 2
    p = gp_Pnt(0, 0, 0)
    v1 = gp_Vec(0, 0, 0)
    result[0].D1(tmid, p, v1)  # handle.GetObject() gives Geom_Curve type, p:gp_Pnt, v1:gp_Vec

    return [p.Coord(), v1.Coord()]


def edge_dihedral(edge, faces):
    """
    Calculate the dihedral angle of an edge
    """
    [midPnt, tangent] = ask_edge_midpnt_tangent(edge)
    uv0 = ask_point_uv2(midPnt, faces[0])
    uv1 = ask_point_uv2(midPnt, faces[1])
    n0 = ask_point_normal_face(uv0, faces[0])
    n1 = ask_point_normal_face(uv1, faces[1])

    if edge.Orientation() == TopAbs_FORWARD:
        cp = np.cross(n0, n1)
        r = np.dot(cp, tangent)
        s = np.sign(r)

    else:
        cp = np.cross(n1, n0)
        r = np.dot(cp, tangent)
        s = np.sign(r)

    return s
