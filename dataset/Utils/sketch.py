import random
import math
import numpy as np

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.gp import gp_Circ, gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.GC import GC_MakeSegment, GC_MakeArcOfCircle
from OCC.Core.Geom import Geom_Circle

import Utils.occ_utils as occ_utils


def triangulation_from_face(face):
    aLoc = TopLoc_Location()
    aTriangulation = BRep_Tool().Triangulation(face, aLoc).GetObject()
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
                vt_map[pid].append(i -1)
            else:
                vt_map[pid] = [i-1]

        edges = [(pids[0], pids[1]), (pids[0], pids[2]), (pids[1], pids[2])]
        for edge in edges:
            if edge in et_map:
                et_map[edge].append(i - 1)
            else:
                et_map[edge] = [i - 1]

    return pts, triangles, vt_map, et_map


def triangles_from_faces(faces):
    tri_list = []
    for face in faces:
        pts, triangles, vt_map, et_map = triangulation_from_face(face)
        for tri in triangles:
            tri_list.append((pts[tri[0]], pts[tri[1]], pts[tri[2]]))

    return tri_list


#==============================================================================
# sketch face generator    
#============================================================================== 
    

def face_circle(ref_pnts):
    dir_w = ref_pnts[2] - ref_pnts[1]
    dir_h = ref_pnts[0] - ref_pnts[1]    
    width = np.linalg.norm(dir_w)
    height = np.linalg.norm(dir_h)
            
    dir_w = dir_w / width
    dir_h = dir_h / height
    normal = np.cross(dir_w, dir_h)
    
    radius = min(width / 2, height / 2)
    
    center = (ref_pnts[0] + ref_pnts[1] + ref_pnts[2] + ref_pnts[3]) / 4

    circ = gp_Circ(gp_Ax2(gp_Pnt(center[0], center[1], center[2]), occ_utils.as_occ(normal, gp_Dir)), radius)
    edge = BRepBuilderAPI_MakeEdge(circ, 0., 2*math.pi).Edge()
    outer_wire = BRepBuilderAPI_MakeWire(edge).Wire()

    face_maker = BRepBuilderAPI_MakeFace(outer_wire)

    return face_maker.Face()

    
def face_circle_1(ref_pnts):
    # sample radius
    edge_dir = ref_pnts[2] - ref_pnts[1]
    width = np.linalg.norm(edge_dir)
    edge_dir = edge_dir / width
    height = np.linalg.norm(ref_pnts[0] - ref_pnts[1])
            
    radius = min(width / 2, height)    
    center = (ref_pnts[1] + ref_pnts[2]) / 2
    if radius < width / 2:
        pnt1 = occ_utils.as_occ(center - edge_dir * radius, gp_Pnt)
        pnt2 = occ_utils.as_occ(center + edge_dir * radius, gp_Pnt)
    else:    
        pnt1 = occ_utils.as_occ(ref_pnts[1], gp_Pnt)
        pnt2 = occ_utils.as_occ(ref_pnts[2], gp_Pnt)
    center = occ_utils.as_occ(center, gp_Pnt)
    
    edge1 = BRepBuilderAPI_MakeEdge(GC_MakeSegment(pnt1, pnt2).Value()).Edge()

    normal = occ_utils.as_occ(np.cross(edge_dir, ref_pnts[0] - ref_pnts[1]), gp_Dir)
    circ = gp_Circ(gp_Ax2(center, normal), radius)
    edge2 = BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(circ, pnt2, pnt1, True).Value()).Edge()

    wire_maker = BRepBuilderAPI_MakeWire(edge1, edge2)
    try:
        face_maker = BRepBuilderAPI_MakeFace(wire_maker.Wire())    
    except RuntimeError as error:
        print(error)
        return None
    return face_maker.Face()

    
def face_circle_2(ref_pnts):
    vec0 = ref_pnts[0] - ref_pnts[1]
    vec2 = ref_pnts[2] - ref_pnts[1]
    width = np.linalg.norm(vec2)
    height = np.linalg.norm(vec0)
            
    radius = min(width, height)
    vec0 = vec0 / np.linalg.norm(vec0)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    pt0 = occ_utils.as_occ(ref_pnts[1] + vec0 * radius, gp_Pnt)
    pt1 = occ_utils.as_occ(ref_pnts[1], gp_Pnt)
    pt2 = occ_utils.as_occ(ref_pnts[1] + vec2 * radius, gp_Pnt)
    
    normal = occ_utils.as_occ(np.cross(vec2, vec0), gp_Dir)
    cir = gp_Circ(gp_Ax2(pt1, normal), radius)    
    seg_maker = [GC_MakeSegment(pt0, pt1), GC_MakeSegment(pt1, pt2), GC_MakeArcOfCircle(cir, pt2, pt0, True)]
    wire_maker = BRepBuilderAPI_MakeWire()
    for sm in seg_maker:
        if sm.IsDone:
            edge = BRepBuilderAPI_MakeEdge(sm.Value()).Edge()
            wire_maker.Add(edge)
        else:
            return None
            
    face_maker = BRepBuilderAPI_MakeFace(wire_maker.Wire())

    return face_maker.Face()

    
def face_circular_end_rect(ref_pnts):
    dir_w = ref_pnts[2] - ref_pnts[1]
    dir_h = ref_pnts[0] - ref_pnts[1]    
    width = np.linalg.norm(dir_w)
    height = np.linalg.norm(dir_h)
    
    pt0 = ref_pnts[0]
    pt1 = ref_pnts[1]
    pt2 = ref_pnts[2]
    pt3 = ref_pnts[3]
    if width < height:
        pt0 = ref_pnts[1]
        pt1 = ref_pnts[2]
        pt2 = ref_pnts[3]
        pt3 = ref_pnts[0]
        radius = width
        width = height
        height = radius
        
    radius = min(height / 2, width / 2 - 0.5)        
    dir_w = (pt2 - pt1) / width
    dir_h = (pt0 - pt1) / height    
    
    pt1 = pt1 + dir_w * radius
    pt2 = pt2 - dir_w * radius
    pt3 = pt2 + dir_h * 2 * radius #TODO(ANDREW): Why does this not use pt3?
    pt0 = pt1 + dir_h * 2 * radius
    c01 = (pt1 + pt0) / 2
    c23 = (pt2 + pt3) / 2
    
    pt1 = occ_utils.as_occ(pt1, gp_Pnt)
    pt0 = occ_utils.as_occ(pt0, gp_Pnt)
    c01 = occ_utils.as_occ(c01, gp_Pnt)
    pt2 = occ_utils.as_occ(pt2, gp_Pnt)
    pt3 = occ_utils.as_occ(pt3, gp_Pnt)
    c23 = occ_utils.as_occ(c23, gp_Pnt)
    
    normal = occ_utils.as_occ(np.cross(dir_w, dir_h), gp_Dir)
    cir01 = gp_Circ(gp_Ax2(c01, normal), radius)
    cir23 = gp_Circ(gp_Ax2(c23, normal), radius)    
    seg_maker = [GC_MakeArcOfCircle(cir01, pt0, pt1, True), GC_MakeSegment(pt1, pt2),
                 GC_MakeArcOfCircle(cir23, pt2, pt3, True), GC_MakeSegment(pt3, pt0)]
    wire_maker = BRepBuilderAPI_MakeWire()
    for sm in seg_maker:
        if sm.IsDone:
            edge = BRepBuilderAPI_MakeEdge(sm.Value()).Edge()
            wire_maker.Add(edge)
        else:
            return None
            
    face_maker = BRepBuilderAPI_MakeFace(wire_maker.Wire())

    return face_maker.Face()

    
def face_open_circular_end_rect_v(ref_pnts):
    dir_w = ref_pnts[2] - ref_pnts[1]
    dir_h = ref_pnts[0] - ref_pnts[1]
    width = np.linalg.norm(dir_w)
    height = np.linalg.norm(dir_h)
    if height - width / 2 > 1.0:
        radius = width / 2
    else:
        radius = random.uniform(0.5, height / 2)
    
    offset = width / 2 - radius
    dir_w = dir_w / width
    dir_h = dir_h / height
    normal = np.cross(dir_w, dir_h)

    pt1 = ref_pnts[1] + dir_w * offset
    pt2 = pt1 + dir_w * 2 * radius
    pt3 = pt2 + dir_h * (height - radius)
    pt4 = pt1 + dir_h * (height - radius)

    center = pt4 + dir_w * radius
    circ = gp_Circ(gp_Ax2(gp_Pnt(center[0], center[1], center[2]), occ_utils.as_occ(normal, gp_Dir)), radius)
    pt1 = occ_utils.as_occ(pt1, gp_Pnt)
    pt2 = occ_utils.as_occ(pt2, gp_Pnt)
    pt3 = occ_utils.as_occ(pt3, gp_Pnt)
    pt4 = occ_utils.as_occ(pt4, gp_Pnt)    
    seg_maker = [GC_MakeSegment(pt1, pt2), GC_MakeSegment(pt2, pt3), GC_MakeArcOfCircle(circ, pt3, pt4, True), GC_MakeSegment(pt4, pt1)]
    wire_maker = BRepBuilderAPI_MakeWire()
    for sm in seg_maker:
        edge = BRepBuilderAPI_MakeEdge(sm.Value()).Edge()
        wire_maker.Add(edge)
            
    face_maker = BRepBuilderAPI_MakeFace(wire_maker.Wire())
    
    return face_maker.Face()

    
def face_open_circular_end_rect_h(ref_pnts):
    dir_w = ref_pnts[2] - ref_pnts[1]
    dir_h = ref_pnts[0] - ref_pnts[1]
    width = np.linalg.norm(dir_w)
    height = np.linalg.norm(dir_h)

    dir_w = dir_w / width
    dir_h = dir_h / height
    normal = np.cross(dir_w, dir_h)

    rect_h = min(height, width / 2 - 0.5)    
    pt1 = ref_pnts[1]
    pt2 = ref_pnts[2]
    pt3 = pt2 - dir_w * rect_h + dir_h * rect_h
    pt4 = pt1 + dir_w * rect_h + dir_h * rect_h
    
    center1 = pt1 + dir_w * rect_h
    center2 = pt2 - dir_w * rect_h
    circ1 = gp_Circ(gp_Ax2(occ_utils.as_occ(center1, gp_Pnt), occ_utils.as_occ(normal, gp_Dir)), rect_h)
    circ2 = gp_Circ(gp_Ax2(occ_utils.as_occ(center2, gp_Pnt), occ_utils.as_occ(normal, gp_Dir)), rect_h)
    pt1 = occ_utils.as_occ(pt1, gp_Pnt)
    pt2 = occ_utils.as_occ(pt2, gp_Pnt)
    pt3 = occ_utils.as_occ(pt3, gp_Pnt)
    pt4 = occ_utils.as_occ(pt4, gp_Pnt)
    
    seg_maker = [GC_MakeSegment(pt1, pt2), GC_MakeArcOfCircle(circ2, pt2, pt3, True), GC_MakeSegment(pt3, pt4), GC_MakeArcOfCircle(circ1, pt4, pt1, True)]
    wire_maker = BRepBuilderAPI_MakeWire()
    for sm in seg_maker:
        edge = BRepBuilderAPI_MakeEdge(sm.Value()).Edge()
        wire_maker.Add(edge)
            
    face_maker = BRepBuilderAPI_MakeFace(wire_maker.Wire())
    
    return face_maker.Face()

    
def face_hexagon(ref_pnts):
    dir_w = ref_pnts[2] - ref_pnts[1]
    dir_h = ref_pnts[0] - ref_pnts[1]    
    width = np.linalg.norm(dir_w)
    height = np.linalg.norm(dir_h)
        
    dir_w = dir_w / width
    dir_h = dir_h / height
    normal = occ_utils.as_occ(np.cross(dir_w, dir_h), gp_Dir)

    radius = min(width / 2, height / 2)
    
    center = (ref_pnts[0] + ref_pnts[1] + ref_pnts[2] + ref_pnts[3]) / 4
    
    circ = Geom_Circle(gp_Ax2(gp_Pnt(center[0], center[1], center[2]), normal), radius)

    ang1 = random.uniform(0.0, math.pi / 3)
    pt1 = occ_utils.as_list(circ.Value(ang1))

    ang2 = ang1 + math.pi / 3
    pt2 = occ_utils.as_list(circ.Value(ang2))

    ang3 = ang2 + math.pi / 3
    pt3 = occ_utils.as_list(circ.Value(ang3))

    ang4 = ang3 + math.pi / 3
    pt4 = occ_utils.as_list(circ.Value(ang4))

    ang5 = ang4 + math.pi / 3
    pt5 = occ_utils.as_list(circ.Value(ang5))

    ang6 = ang5 + math.pi / 3
    pt6 = occ_utils.as_list(circ.Value(ang6))

    return occ_utils.face_polygon([pt1, pt2, pt3, pt4, pt5, pt6])

    
def face_oring(ref_pnts):
    dir_w = ref_pnts[2] - ref_pnts[1]
    dir_h = ref_pnts[0] - ref_pnts[1]    
    width = np.linalg.norm(dir_w)
    height = np.linalg.norm(dir_h)
    
    dir_w = dir_w / width
    dir_h = dir_h / height
    normal = occ_utils.as_occ(np.cross(dir_w, dir_h), gp_Dir)

    outer_r = min(width / 2, height / 2)    
    center = (ref_pnts[0] + ref_pnts[1] + ref_pnts[2] + ref_pnts[3]) / 4
    
    inner_r= random.uniform(outer_r / 3, outer_r - 0.2)

    circ = gp_Circ(gp_Ax2(gp_Pnt(center[0], center[1], center[2]), normal), outer_r)
    edge = BRepBuilderAPI_MakeEdge(circ, 0., 2*math.pi).Edge()
    outer_wire = BRepBuilderAPI_MakeWire(edge).Wire()

    normal.Reverse()
    circ = gp_Circ(gp_Ax2(gp_Pnt(center[0], center[1], center[2]), normal), inner_r)
    edge = BRepBuilderAPI_MakeEdge(circ, 0., 2*math.pi).Edge()
    inner_wire = BRepBuilderAPI_MakeWire(edge).Wire()

    face_maker = BRepBuilderAPI_MakeFace(outer_wire)
    face_maker.Add(inner_wire)

    return face_maker.Face()

    
def face_pentagon(ref_pnts):
    dir_l = ref_pnts[0] - ref_pnts[1]
    dir_r = ref_pnts[3] - ref_pnts[2]    

    ratio = random.uniform(0.4, 0.8)
    pt4 = (ref_pnts[0] + ref_pnts[3]) / 2    
    pt0 = ref_pnts[1] +  dir_l * ratio
    pt1 = ref_pnts[1]
    pt2 = ref_pnts[2]
    pt3 = ref_pnts[2] + dir_r * ratio
    
    return occ_utils.face_polygon([pt0, pt1, pt2, pt3, pt4])
    
    
def face_quad(ref_pnts):
    dir_l = ref_pnts[0] - ref_pnts[1]
    dir_r = ref_pnts[3] - ref_pnts[2]    
        
    mark = [0, 1]
    random.shuffle(mark)
    ratio = random.uniform(0.3, 0.6)
    pt0 = ref_pnts[0] - dir_l * mark[0] * ratio
    pt1 = ref_pnts[1]
    pt2 = ref_pnts[2]
    pt3 = ref_pnts[3] - dir_r * mark[1] * ratio

    return occ_utils.face_polygon([pt0, pt1, pt2, pt3])
  
    
def face_rect(ref_pnts):
    return occ_utils.face_polygon(ref_pnts[:4])

    
def face_triangle(ref_pnts):
    dir_w = ref_pnts[2] - ref_pnts[1]
    dir_h = ref_pnts[0] - ref_pnts[1]    
    width = np.linalg.norm(dir_w)
    height = np.linalg.norm(dir_h)
        
    dir_w = dir_w / width
    dir_h = dir_h / height
    normal = occ_utils.as_occ(np.cross(dir_w, dir_h), gp_Dir)
    
    radius = min(width / 2, height / 2)
    
    center = (ref_pnts[0] + ref_pnts[1] + ref_pnts[2] + ref_pnts[3]) / 4

    circ = Geom_Circle(gp_Ax2(gp_Pnt(center[0], center[1], center[2]), normal), radius)
    
    ang1 = random.uniform(0.0, 2 * math.pi / 3)
    pt1 = occ_utils.as_list(circ.Value(ang1))

    ang2 = ang1 + random.uniform(2 * math.pi / 3 - math.pi / 9, 2 * math.pi / 3 + math.pi / 9)
    if ang2 > 2 * math.pi:
        ang2 = ang2 - 2 * math.pi
    pt2 = occ_utils.as_list(circ.Value(ang2))

    ang3 = ang2 + random.uniform(2 * math.pi / 3 - math.pi / 9, 2 * math.pi / 3 + math.pi / 9)
    if ang3 > 2 * math.pi:
        ang3 = ang3 - 2 * math.pi
    pt3 = occ_utils.as_list(circ.Value(ang3))

    return occ_utils.face_polygon([pt1, pt2, pt3])

    
def face_triangle_1(ref_pnts):
    edge_dir = ref_pnts[2] - ref_pnts[1]

    normal = np.cross(edge_dir, ref_pnts[0] - ref_pnts[1])
    edge_normal = np.cross(normal, edge_dir)
    edge_normal = edge_normal / np.linalg.norm(edge_normal)
    pnt3 = ref_pnts[0] + edge_dir / 2

    return occ_utils.face_polygon([ref_pnts[1], ref_pnts[2], pnt3])

    
def face_triangle_2(ref_pnts):     
    return occ_utils.face_polygon([ref_pnts[0], ref_pnts[1], ref_pnts[2]])
