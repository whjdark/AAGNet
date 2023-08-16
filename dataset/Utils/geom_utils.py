# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:03:49 2019

@author: 2624224
"""
import math
import numpy as np


def ray_triangle_intersect(ray_tri):
    '''
    https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
    input:
        ray_tri: ([float, float, float],) * 5
    output:
        float
    '''
    ray_origin, ray_direction, tri_v0, tri_v1, tri_v2 = ray_tri

    ray_origin = np.asarray(ray_origin)
    ray_direction = np.asarray(ray_direction)
    tri_v0 = np.asarray(tri_v0)
    tri_v1 = np.asarray(tri_v1)
    tri_v2 = np.asarray(tri_v2)

    v0v1 = tri_v1 - tri_v0
    v0v2 = tri_v2 - tri_v0
    pvec = np.cross(ray_direction, v0v2)

    det = np.dot(v0v1, pvec)

    if abs(det) < 0.000001:        
        return float('-inf')

    invDet = 1.0 / det
    tvec = ray_origin - tri_v0
    u = np.dot(tvec, pvec) * invDet

    if u < 0 or u > 1:        
        return float('-inf')

    qvec = np.cross(tvec, v0v1)
    v = np.dot(ray_direction, qvec) * invDet

    if v < 0 or u + v > 1:
        return float('-inf')

    return np.dot(v0v2, qvec) * invDet


def ray_triangle_set_intersect(ray_origin, ray_direction, tri_list):
    '''
    input:
        ray_origin: [float, float, float] * n
        ray_direction: [float, float, float] * n
        tri_list: [[[float, float, float] * 3]] * n
    output:
        float
    '''
    ray_tri_list = []
    for tri in tri_list:
        ray_tri_list.append((ray_origin, ray_direction, tri[0], tri[1], tri[2]))

    results = []
    for ray_tri in ray_tri_list:
        results.append(ray_triangle_intersect(ray_tri))
#    results = Pool().map(ray_triangle_intersect, ray_tri_list)
    results = np.asarray(results)
    results = results[results > 0]
    if np.size(results) == 0:        
        #print('ray no intersect', ray_origin, ray_direction)
        return float('-inf')
        
    return min(results)


def point_in_polygon(the_pnt, verts, closed=True, normal=None):
    num_v = len(verts)
    assert num_v > 1
    
    if num_v == 2:
        assert normal is not None
        
    if num_v > 2:
        vec1 = verts[1] - verts[0]
        vec2 = verts[2] - verts[1]
        normal = np.cross(vec1, vec2)

    if closed:
        verts = np.append(verts, [verts[0]], axis=0)      

    dists = np.array([dist_pnt_line(the_pnt, verts[i], verts[i + 1], normal) for i in range(len(verts) - 1)])
    if np.sum(dists <= 0.000001) > 0:
        return False
    else:        
        return True

    
def point_in_polygon_set(the_pnt, polygons):
    for poly in polygons:
        if point_in_polygon(the_pnt, poly) == True:
            return True
    
    return False

    
def points_in_polygon(bnd_pnts, verts, closed=True, normal=None):
    idx = np.array([point_in_polygon(pnt, verts, closed, normal) for pnt in bnd_pnts])
    in_pnts = bnd_pnts[idx]
    
    return in_pnts


def dist_pnt_line(pnt, pnt1, pnt2, normal):
    line_dir = pnt2 - pnt1
    pnt_dir = pnt - pnt1
    perp_dir = np.cross(normal, line_dir)
    perp_dir = perp_dir / np.linalg.norm(perp_dir)
    return np.dot(pnt_dir, perp_dir)

    
def ray_segment_intersect(ray_pnt, ray_dir, pnt1, pnt2):
    thres = 0.000001
    pnt1 = np.array(pnt1)
    pnt2 = np.array(pnt2)
    seg_dir = pnt2 - pnt1
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    
    # check if ray origin lie on segment
    vec1 = pnt1 - ray_pnt
    vec2 = pnt2 - ray_pnt
    origin_on_segment = np.linalg.norm(np.cross(vec1, vec2)) < thres
    normal = np.cross(seg_dir, ray_dir)
    if origin_on_segment: 
        if np.dot(vec1, vec2) < thres:
            return 0.0
        else:
            if np.linalg.norm(normal) < thres:
                dist1 = np.dot(vec1, ray_dir)
                dist2 = np.dot(vec2, ray_dir)
                if dist1 > thres and dist2 > thres:
                    if dist1 < dist2:
                        return np.linalg.norm(vec1)
                    else:
                        return np.linalg.norm(vec2)
            else:
                return float('-inf')
            
    # check if ray and segment are parallel
    if np.linalg.norm(normal) < thres:
        return float('-inf')

    # check if ray lie on one side of segment
    if np.dot(vec1, ray_dir) < 0 and np.dot(vec2, ray_dir) < 0:
        return float('-inf')
    
    # check if segment lie on one side of ray
    if np.dot(np.cross(vec1, ray_dir), np.cross(vec2, ray_dir)) > 0:        
        return float('-inf')            
    
        
    seg_normal = np.cross(normal, seg_dir)
    seg_normal = seg_normal / np.linalg.norm(seg_normal)
    dist = np.dot(vec1, seg_normal) / np.dot(ray_dir, seg_normal)
    
    try:
        assert not math.isnan(dist), 'dist is not a number'
    except AssertionError as error:
        print('ray_segment_intersect', error)
        print(ray_pnt, ray_dir, pnt1, pnt2)
        return float('-inf')
    return dist

    
def ray_segment_set_intersect(ray_pnt, ray_dir, segs):
    intersects = []
    for seg in segs:
        int_pnt = ray_segment_intersect(ray_pnt, ray_dir, seg[0], seg[1])
        if int_pnt >= 0.0:
            intersects.append(int_pnt)
            
    return intersects
    
    
def search_rect_inside_bound_2(pnt1, vec0, vec2, bnd_pnts):
    verts = np.array([pnt1 + vec0, pnt1, pnt1 + vec2, pnt1 + vec0 + vec2])      
    in_pnts = points_in_polygon(bnd_pnts, verts)
    if len(in_pnts) == 0:        
        return verts

    vec0_len = np.linalg.norm(vec0)
    vec0 = vec0 / vec0_len
    vec2_len = np.linalg.norm(vec2)
    vec2 = vec2 / vec2_len
    len2 = min([np.dot(pnt - pnt1, vec2) for pnt in in_pnts] + [vec2_len])
    len0 = min([np.dot(pnt - pnt1, vec0) for pnt in in_pnts] + [vec0_len])
    
    vec0 = vec0 * len0 
    vec2 = vec2 * len2     
    
    verts[0] = verts[1] + vec0
    verts[2] = verts[1] + vec2
    verts[3] = verts[1] + vec0 + vec2

    try:
        for vert in verts:
            assert not math.isnan(vert[0]), 'vert[0] is not a number'
            assert not math.isnan(vert[1]), 'vert[1] is not a number'
            assert not math.isnan(vert[2]), 'vert[2] is not a number'
    except AssertionError as error:
        print('search_rect_inside_bound_2', error)
        print(pnt1, vec0, vec2)
        return None
        
    return verts


def search_rect_inside_bound_3(pnt1, pnt2, vec1, vec2, bnd_pnts):
    verts = np.array([pnt1 + vec1, pnt1, pnt2, pnt2 + vec2])
    in_pnts = points_in_polygon(bnd_pnts, verts)
    if len(in_pnts) == 0:
      return verts
            
    normal = np.cross(vec2, pnt1 - pnt2)
    line_dir = pnt2 - pnt1
    perp_dir = np.cross(normal, line_dir)
    perp_dir = perp_dir / np.linalg.norm(perp_dir)

    norm1 = np.dot(vec1, perp_dir)
    norm2 = np.dot(vec2, perp_dir)    
    dist = min([np.dot(pnt - pnt1, perp_dir) for pnt in in_pnts] + [norm1, norm2])
    vec1 = vec1 * dist / norm1
    vec2 = vec2 * dist / norm2
     
    verts[0] = verts[1] + vec1
    verts[3] = verts[2] + vec2

    return verts


def points_inside_rect(pnt0, pnt1, pnt2, pnt3, resolution=0.5):
    pnt0 = np.array(pnt0)
    pnt1 = np.array(pnt1)
    pnt2 = np.array(pnt2)
    pnt3 = np.array(pnt3)
    
    dir_w = pnt2 - pnt1
    dir_h = pnt0 - pnt1
    width = np.linalg.norm(dir_w)
    height = np.linalg.norm(dir_h)
    num_w = int(width / resolution)
    if num_w == 0:
        num_w = 1
    num_h = int(height / resolution)
    if num_h == 0:
        num_h = 1
    delta_w = width / num_w
    delta_h = height / num_h
    
    dir_w = dir_w / width
    dir_h = dir_h / height
    
    idx = np.array(np.meshgrid(np.arange(0, num_w + 1), np.arange(0, num_h + 1))).T.reshape(-1,2)   
    points = pnt1 + dir_w * delta_w * np.array(idx[:, 0]).reshape(-1, 1) + dir_h * delta_h * np.array(idx[:, 1]).reshape(-1, 1)
    return points


def dist_point_plane(pnt, pl_pnt, pl_normal):
    p_dir = np.array(pnt) - np.array(pl_pnt)    
    
    dist = np.dot(p_dir, np.array(pl_normal))
    
    return dist


def outer_radius_triangle(pt1, pt2, pt3):
    a = np.linalg.norm(np.array(pt1) - np.array(pt2))
    b = np.linalg.norm(np.array(pt2) - np.array(pt3))
    c = np.linalg.norm(np.array(pt3) - np.array(pt1))
    p = (a + b + c) / 2
    return a * b * c / (4 * np.sqrt(p * (p - a) * (p - b) * (p - c)))
