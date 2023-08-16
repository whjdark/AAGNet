# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:57:43 2018

@author: 2624224
"""

from math import pi
import random

from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_Transform, BRepBuilderAPI_MakeWire,
                                BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace)
from OCC.Core.BRepFeat import BRepFeat_MakePrism
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir, gp_Ax1, gp_Trsf, gp_Vec, gp_OZ, gp_Circ
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.GC import GC_MakeArcOfCircle, GC_MakeSegment
from OCC.Core.TopTools import TopTools_ListIteratorOfListOfShape

from OCC.Core.gp import gp_Pnt
from OCC.Core.BRep import BRep_Tool
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.TopoDS import topods_Face
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED

import Utils.occ_utils as occ_utils

DRAIN_R = 10.0
DRAIN_S = 0.5
DRAIN_T = 1.0
DRAIN_RCS = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))


def wire_circle():
    '''
        standard circle on XY plane, centered at origin, with radius 1

    output
        w:     TopoDS_Wire
    '''
    circ = gp_Circ(DRAIN_RCS, 1.0)
    edge = BRepBuilderAPI_MakeEdge(circ, 0., 2*pi).Edge()
    wire = BRepBuilderAPI_MakeWire(edge).Wire()

    return wire


def wire_triangle3():
    '''
    equal sided triangle, centered at origin
    output
        w:  TopoDS_Wire
    '''
    pt1 = gp_Pnt(-1, 0, 0)
    pt2 = gp_Pnt(-1, 0, 0)
    pt2.Rotate(gp_OZ(), 2*pi/3)
    pt3 = gp_Pnt(-1, 0, 0)
    pt3.Rotate(gp_OZ(), 4*pi/3)

    ed1 = BRepBuilderAPI_MakeEdge(GC_MakeSegment(pt1, pt2).Value()).Edge()
    ed2 = BRepBuilderAPI_MakeEdge(GC_MakeSegment(pt2, pt3).Value()).Edge()
    ed3 = BRepBuilderAPI_MakeEdge(GC_MakeSegment(pt3, pt1).Value()).Edge()

    wire = BRepBuilderAPI_MakeWire(ed1, ed2, ed3).Wire()

    return wire


def wire_triangle2():
    '''
        isosceles triangle
    output
        w:  TopoDS_Wire
    '''
    ang = random.gauss(2*pi/3, pi/6)
    amin = pi / 3
    amax = 5 * pi / 6
    if ang > amax:
        ang = amax
    if ang < amin:
        ang = amin
    pt1 = gp_Pnt(-1, 0, 0)
    pt2 = gp_Pnt(-1, 0, 0)
    pt2.Rotate(gp_OZ(), ang)
    pt3 = gp_Pnt(-1, 0, 0)
    pt3.Rotate(gp_OZ(), -ang)

    ed1 = BRepBuilderAPI_MakeEdge(GC_MakeSegment(pt1, pt2).Value()).Edge()
    ed2 = BRepBuilderAPI_MakeEdge(GC_MakeSegment(pt2, pt3).Value()).Edge()
    ed3 = BRepBuilderAPI_MakeEdge(GC_MakeSegment(pt3, pt1).Value()).Edge()

    wire = BRepBuilderAPI_MakeWire(ed1, ed2, ed3).Wire()

    return wire


def wire_rectangle():
    '''
    output
        w:  TopoDS_Wire
    '''
    pt1 = gp_Pnt(0, 1, 0)
    pt2 = gp_Pnt(-1, 0, 0)
    pt3 = gp_Pnt(0, -1, 0)
    pt4 = gp_Pnt(1, 0, 0)

    ed1 = BRepBuilderAPI_MakeEdge(GC_MakeSegment(pt1, pt2).Value()).Edge()
    ed2 = BRepBuilderAPI_MakeEdge(GC_MakeSegment(pt2, pt3).Value()).Edge()
    ed3 = BRepBuilderAPI_MakeEdge(GC_MakeSegment(pt3, pt4).Value()).Edge()
    ed4 = BRepBuilderAPI_MakeEdge(GC_MakeSegment(pt4, pt1).Value()).Edge()

    wire = BRepBuilderAPI_MakeWire(ed1, ed2, ed3, ed4).Wire()

    return wire


def wire_sweep_circle(ct1, ct2):
    '''
    input
        c1:     gp_Pnt
        c2:     gp_Pnt
    output
        w:      TopoDS_Wire
    '''
    center = DRAIN_RCS.Location()
    vec = DRAIN_RCS.Direction()

    radius = center.Distance(ct1)

    pt1 = gp_Pnt(ct1.XYZ())
    pt2 = gp_Pnt(ct1.XYZ())
    pt3 = gp_Pnt(ct2.XYZ())
    pt4 = gp_Pnt(ct2.XYZ())

    vec1 = gp_Vec(ct1, center)
    vec1.Normalize()
    vec2 = gp_Vec(ct2, center)
    vec2.Normalize()

    pt1.Translate(vec1*DRAIN_S)
    pt2.Translate(-vec1*DRAIN_S)
    pt3.Translate(vec2*DRAIN_S)
    pt4.Translate(-vec2*DRAIN_S)

    cir1 = gp_Circ(gp_Ax2(ct1, vec), DRAIN_S)
    ed1 = BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(cir1, pt1, pt2, True).Value()).Edge()

    cir2 = gp_Circ(gp_Ax2(center, vec), radius + DRAIN_S)
    ed2 = BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(cir2, pt2, pt4, False).Value()).Edge()

    cir3 = gp_Circ(gp_Ax2(ct2, vec), DRAIN_S)
    ed3 = BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(cir3, pt4, pt3, True).Value()).Edge()

    cir4 = gp_Circ(gp_Ax2(center, vec), radius - DRAIN_S)
    ed4 = BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(cir4, pt1, pt3, False).Value()).Edge()

    wire = BRepBuilderAPI_MakeWire(ed1, ed2, ed3, ed4).Wire()

    return wire


# list of wire generation function
FLIST = [wire_circle, wire_rectangle, wire_triangle2, wire_sweep_circle]
SKETCH_TYPE = ['circle', 'rectangle', 'triangle2', 'sweep']
FEAT_TYPE = ['hole', 'blind', 'boss']
LABEL_INDEX = {'other': 0, 'base': 1, 'hole_triangle2': 2, 'hole_rectangle': 3, 'hole_circle': 4,
               'hole_sweep': 5, 'blind_triangle2': 6, 'blind_rectangle': 7, 'blind_circle': 8,
               'blind_sweep': 9, 'boss_triangle2': 10, 'boss_rectangle': 11, 'boss_circle': 12,
               'boss_sweep': 13}


def len_seq_natural(pos, pos_list):
    '''
        find the length of the natural sequence from pos, pos is an element of pos_list
    input
        pos:        int
        pos_list:   [int]
    output
        j - i:      int
    '''
    i = pos_list.index(pos)
    j = i + 1
    while j < len(pos_list):
        if pos_list[j] != pos_list[j - 1] + 1:
            break
        j += 1
    return j - i


def list_wire_combo(num_cell, ang, offset, radius):
    '''
    input
       nc:              int, number of cells to be combined
       ang:             float, angle between adjaent cells
       offset:          float, offset angle of start position
       ri:              float, radius of this ring
    output
        wlist:          {TopoDS_Wire: string}
        combo_name:     ''
    '''
    combo_name = ''
    pos_list = list(range(num_cell))
    wlist = {}
    pos_len_name = {}
    while len(pos_list) > 0:
#       1 choose a random location
        pos = random.choice(pos_list)

#       2 choose a random length
        len_seq = len_seq_natural(pos, pos_list)
        len_seq = random.randrange(1, len_seq + 1)

#       3 choose a random shape
        func = random.choice(FLIST)
#        print(pos_list, pos, l, fname[FLIST.index(func)])
        trsf_scale = gp_Trsf()
        trsf_scale.SetScale(DRAIN_RCS.Location(), DRAIN_S)
        trsf_trans = gp_Trsf()
        trans_vec = gp_Vec(DRAIN_RCS.XDirection()) * radius
        trsf_trans.SetTranslation(trans_vec)
        trsf_rotate = gp_Trsf()
        trsf_rotate.SetRotation(gp_Ax1(DRAIN_RCS.Location(), DRAIN_RCS.Direction()),
                                offset + pos * ang)
        if func == wire_sweep_circle and len_seq > 1:
            cir1 = DRAIN_RCS.Location()
            cir2 = DRAIN_RCS.Location()
            cir1.Translate(trans_vec)
            cir1.Rotate(gp_Ax1(DRAIN_RCS.Location(), DRAIN_RCS.Direction()), offset + pos * ang)
            cir2.Translate(trans_vec)
            cir2.Rotate(gp_Ax1(DRAIN_RCS.Location(), DRAIN_RCS.Direction()),
                        offset + (pos + len_seq -1) * ang)
            wire = wire_sweep_circle(cir1, cir2)
        elif func != wire_sweep_circle and len_seq == 1:
            wire = func()
            bresp_trsf = BRepBuilderAPI_Transform(wire, trsf_scale)
            wire = topods.Wire(bresp_trsf.Shape())
            bresp_trsf = BRepBuilderAPI_Transform(wire, trsf_trans)
            wire = topods.Wire(bresp_trsf.Shape())
            bresp_trsf = BRepBuilderAPI_Transform(wire, trsf_rotate)
            wire = topods.Wire(bresp_trsf.Shape())
        else:
            continue

        wname = SKETCH_TYPE[FLIST.index(func)]
        pos_len_name[pos] = (len_seq, wname)
        wlist[wire] = wname
        for pos in range(pos, pos + len_seq):
            pos_list.remove(pos)

    pos_len_name = sorted(pos_len_name.items(), key=lambda t: t[0])
    for pos in pos_len_name:
        combo_name += str(pos[1][0]) + '[' + pos[1][1] + ']'
    return wlist, combo_name


def list_wire_random():
    '''
    output
        wires:      {TopoDS_Wire:string}
        wire_name:  ''
    '''
    wire_name = ''
    #    number of rings
    numr = int((DRAIN_R/4/DRAIN_S-0.5))
    wires = {}

    for i in range(numr):
#        radius of ith ring
        radius = 3*DRAIN_S+i*4*DRAIN_S
#        number of cells per ring
        nump = int(1.5*pi+2*pi*i)
#        print('np:',np)

#        randomly choose the number of cells to combine
        combo_list = range(1, nump // 3 + 1)
        combo = random.choice(combo_list)
#        angle between two adjacent cells
        ang = 2 * pi / nump
#        randomly offset the start cell
        offset = random.gauss(ang / 2, ang / 2)
        if offset < 0.:
            offset = 0.
        if offset > ang:
            offset = ang
        wlist, combo_name = list_wire_combo(combo, ang, offset, radius)
        wires.update(wlist)
        wire_name += str(combo) + '(' + combo_name + ')'
        nump = nump // combo
#        print('combo',combo,'repeat',np)

        ang = 2*pi/nump
        for j in range(1, nump):
            trsf = gp_Trsf()
            trsf.SetRotation(gp_Ax1(DRAIN_RCS.Location(), DRAIN_RCS.Direction()), ang * j)
            for wire in wlist:
                wname = wlist[wire]
                bresp_trsf = BRepBuilderAPI_Transform(wire, trsf)
                wire = topods.Wire(bresp_trsf.Shape())
                wires[wire] = wname

    return wires, wire_name


def face_bottom(shape):
    '''
    input
        s: TopoDS_Shape
    output
        f: TopoDS_Face
    '''
    f_list = occ_utils.list_face(shape)
    face = None
    for face in f_list:
        normal = occ_utils.normal_to_face_center(face)
        if normal.IsEqual(DRAIN_RCS.Direction(), 0.01):
            break

    return face


def map_face_before_and_after_feat(base, feature_maker):
    '''
    input
        base: TopoDS_Shape
        feature_maker: BRepFeat_MakePrism
    output
        fmap: {TopoDS_Face:TopoDS_Face}
    '''

    fmap = {}
    base_faces = occ_utils.list_face(base)

    for face in base_faces:
        if feature_maker.IsDeleted(face):
            continue

        fmap[face] = []

        modified = feature_maker.Modified(face)
        if modified.Size() == 0:
            fmap[face].append(face)
            continue

        occ_it = TopTools_ListIteratorOfListOfShape(modified)
        while occ_it.More():
            a_shape = occ_it.Value()
            assert a_shape.ShapeType() == TopAbs_FACE
            fmap[face].append(topods.Face(a_shape))
            occ_it.Next()
        
    return fmap


def map_from_name(shape, name):
    '''
    input
        shape: TopoDS_Shape
        name: string
    output
        name_map: {TopoDS_Face: int}
    '''
    name_map = {}
    faces = occ_utils.list_face(shape)

    for one_face in faces:
        name_map[one_face] = name

    return name_map


def same_shape_in_list(the_shape, slist):
    the_hash = the_shape.__hash__()
    for a_shape in slist:
        if the_hash == a_shape.__hash__():#the_shape.IsSame(a_shape):
            return a_shape
    return None

    
def map_from_shape_and_name(fmap, old_labels, new_shape, new_name, feature_dir=None):
    '''
    input
        fmap: {TopoDS_Face: TopoDS_Face},
        old_map: {TopoDS_Face: int}
        new_shape: TopoDS_Shape
        new_name: string
    output
        new_map:
    '''
    new_map = {}
    new_bottom_label = {}

    if isinstance(old_labels, dict):
        # first feature made
        seg_map = old_labels
        ins_label = []
        bottom_map = {}
        # first stock made, all are not bottom face
        for face in seg_map.keys():
            bottom_map[face] = 0
    elif isinstance(old_labels, tuple):
        seg_map = old_labels[0]
        ins_label = old_labels[1]
        bottom_map = old_labels[2]
    else:
        assert False, 'Invalid map type: %s' % type(old_labels)

    new_faces = occ_utils.list_face(new_shape)
    new_faces_backup = occ_utils.list_face(new_shape)
    
    # after making, some original faces has been modified
    for oldf in fmap:
        old_seg_name = seg_map[oldf]
        old_bottom_name = bottom_map[oldf]
        for samef in fmap[oldf]:            
            samef = same_shape_in_list(samef, new_faces)            
            if samef is None:
                print('no same face')
                continue
            # update segmantic label
            new_map[samef] = old_seg_name
            # update bottom face label
            new_bottom_label[samef] = old_bottom_name
            new_faces.remove(samef)
    
    # new added faces are belong to new feature
    for n_face in new_faces:
        new_map[n_face] = new_name
        # determine machning feature fead direction
        # the normal vector of bottom face is parallel to the machning fead direction
        if feature_dir:
            centroid = ask_face_centroid(n_face)
            uv = ask_point_uv2(centroid, n_face)
            norm_vec = ask_point_normal_face(uv, n_face)
            norm_vec = occ_utils.as_occ(norm_vec, gp_Dir)
            isParallel = feature_dir.IsParallel(norm_vec, 1e-6)
            new_bottom_label[n_face] = int(isParallel)
        else: # no direction feature, such as charmder and round
            new_bottom_label[n_face] = 0

    # update instance label, after making, some original faces has been removed
    if len(ins_label) != 0:
        for ins_idx in range(len(ins_label)):
            new_inst = []
            for old_face in ins_label[ins_idx]:
                if old_face not in fmap:
                    print('mssing old face, which may be deleted')
                    continue
                for same_face in fmap[old_face]:            
                    same_face = same_shape_in_list(same_face, new_faces_backup)            
                    if same_face is None:
                        print('no same face')
                        continue
                    new_inst.append(same_face)
            ins_label[ins_idx] = new_inst
    # add new instance faces
    ins_label.append(new_faces)

    return new_map, ins_label, new_bottom_label


def shape_multiple_hole_feats(base, wlist):
    '''
        one face and one hole feature for each wire
    input
        base:       TopoDS_Shape
        wlist:      {TopoDS_Wire:string}
    output
        base:       TopoDS_Shape
        name_map:   {TopoDS_Face:int}
        ftype:      ''
    '''
    b_face = face_bottom(base)
    random.shuffle(FEAT_TYPE)
    ftype = random.choice(FEAT_TYPE)
    if ftype == 'hole':
        direction = DRAIN_RCS.Direction()
        fuse = False
        length = DRAIN_T
    elif ftype == 'blind':
        direction = DRAIN_RCS.Direction()
        fuse = False
        length = DRAIN_T / 2
    else:
        direction = -DRAIN_RCS.Direction()
        fuse = True
        length = DRAIN_T / 2

    base_map = map_from_name(base, LABEL_INDEX['base'])
    for wire in wlist:
        face_p = BRepBuilderAPI_MakeFace(wire).Face()
        feature_maker = BRepFeat_MakePrism()
        feature_maker.Init(base, face_p, b_face, direction, fuse, False)
        feature_maker.Build()

        feature_maker.Perform(length)
        shape = feature_maker.Shape()

        fmap = map_face_before_and_after_feat(base, feature_maker)
        name_map = map_from_shape_and_name(fmap, base_map, shape,
                                           LABEL_INDEX[ftype + '_' + wlist[wire]])

        base = shape
        base_map = name_map

    return base, base_map, ftype


def shape_base_drain():
    '''
    output
        s: TopoDS_Shape
    '''
    wire = wire_circle()

    trsf = gp_Trsf()
    trsf.SetScale(DRAIN_RCS.Location(), DRAIN_R)
    bresp_trsf = BRepBuilderAPI_Transform(wire, trsf)
    wire = topods.Wire(bresp_trsf.Shape())

    base_face = BRepBuilderAPI_MakeFace(wire).Face()
    shape = BRepPrimAPI_MakePrism(base_face, gp_Vec(0, 0, DRAIN_T)).Shape()

    return shape


def shape_drain():
    '''
    output
        shape:          TopoDS_Shape
        face_map:       {TopoDS_Face: int}
        id_map:         {TopoDS_Face: int}
        shape_name:     ''
    '''
#    print('shape_drain')
    random.seed()
#    step1, create the base
    base = shape_base_drain()

#    step2, create wires for holes
    wlist, wire_name = list_wire_random()

#    step3, add hole feature from wire
    shape, name_map, feat_name = shape_multiple_hole_feats(base, wlist)

    shape_name = feat_name + '-' + wire_name

    fid = 0
    fset = occ_utils.list_face(shape)
    id_map = {}
    for shape_face in fset:
        id_map[shape_face] = fid
        fid += 1

    return shape, name_map, id_map, shape_name


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
