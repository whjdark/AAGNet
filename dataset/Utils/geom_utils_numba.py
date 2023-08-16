import math
import numpy as np
import numba as nb
import Utils.numba_vec as nbv


@nb.njit(fastmath=True)
def search_rect_inside_bound_1(verts, vec1, vec2, bnd_pnts):
    in_pnts = points_in_polygon(bnd_pnts, verts)
    if in_pnts.shape[0] == 0:
        return verts

    # find perpendicular unit vector of normal and line
    normal = nbv.cross(vec2, verts[1] - verts[2])
    line_dir = nbv.sub(verts[1], verts[2])
    perp_dir = nbv.cross(normal, line_dir)
    perp_dir = nbv.div(nbv.calc_l2_norm(perp_dir), perp_dir)

    # Projecting vec1 onto unit_perp_dir
    norm1 = nbv.dot(vec1, perp_dir)
    norm2 = nbv.dot(vec2, perp_dir)
    dist = min([np.dot(pnt - verts[1], perp_dir) for pnt in in_pnts] + [norm1, norm2])
    vec1 = nbv.div(norm1, nbv.mul(dist, vec1))
    vec2 = nbv.div(norm2, nbv.mul(dist, vec2))

    verts[0] = nbv.add(verts[1], vec1)
    verts[3] = nbv.add(verts[2], vec2)

    return verts


@nb.njit(fastmath=True)
def search_rect_inside_bound_2(verts, vec0, vec2, bnd_pnts):
    in_pnts = points_in_polygon(bnd_pnts, verts)

    if in_pnts.shape[0] == 0:
        return verts

    vec0_len = nbv.calc_l2_norm(vec0)
    vec0 = nbv.div(vec0_len, vec0)
    vec2_len = nbv.calc_l2_norm(vec2)
    vec2 = nbv.div(vec2_len, vec2)
    len2 = min([nbv.dot(pnt - verts[1], vec2) for pnt in in_pnts] + [vec2_len])
    len0 = min([nbv.dot(pnt - verts[1], vec0) for pnt in in_pnts] + [vec0_len])

    vec0 = nbv.mul(len0, vec0)
    vec2 = nbv.mul(len2, vec2)

    verts[0] = nbv.add(vec0, verts[1])
    verts[2] = nbv.add(vec2, verts[1])
    verts[3] = nbv.add(nbv.add(vec0, vec2), verts[1])

    for vert in verts:
        assert not math.isnan(vert[0]), 'vert[0] is not a number'
        assert not math.isnan(vert[1]), 'vert[1] is not a number'
        assert not math.isnan(vert[2]), 'vert[2] is not a number'

    return verts


@nb.njit(fastmath=True)
def search_rect_inside_bound_3(verts, bnd_pnts):
    vec1 = nbv.sub(verts[1], verts[0])
    vec2 = nbv.sub(verts[2], verts[3])

    in_pnts = points_in_polygon(bnd_pnts, verts)
    if in_pnts.shape[0] == 0:
        return verts

    # Find perpendicular unit vector of normal and line
    normal = nbv.cross(vec2, verts[1] - verts[2])
    line_dir = nbv.sub(verts[1], verts[2])
    perp_dir = nbv.cross(normal, line_dir)
    perp_dir = nbv.div(nbv.calc_l2_norm(perp_dir), perp_dir)

    # Projecting vec1 onto unit_perp_dir
    norm1 = nbv.dot(vec1, perp_dir)
    norm2 = nbv.dot(vec2, perp_dir)
    dist = min([nbv.dot(nbv.sub(verts[1], pnt), perp_dir) for pnt in in_pnts] + [norm1, norm2])
    vec1 = nbv.div(norm1, nbv.mul(dist, vec1))
    vec2 = nbv.div(norm2, nbv.mul(dist, vec2))

    verts[0] = nbv.add(verts[1], vec1)
    verts[3] = nbv.add(verts[2], vec2)

    return verts


@nb.njit(fastmath=True)
def ray_triangle_set_intersect(ray_origin, ray_direction, tri_list):
    '''
    input:
        ray_origin: [float, float, float] * n
        ray_direction: [float, float, float] * n
        tri_list: [[[float, float, float] * 3]] * n
    output:
        float
    '''
    results = np.zeros(shape=(tri_list.shape[0]))
    for i in range(tri_list.shape[0]):
        results[i] = ray_triangle_intersect(ray_origin, ray_direction, tri_list[i][0], tri_list[i][1], tri_list[i][2])

    results = results[results > 0]

    if results.shape[0] == 0:
        return np.NINF

    return min(results)


@nb.njit(fastmath=True)
def ray_triangle_intersect(ray_origin, ray_direction, tri_v0, tri_v1, tri_v2):
    '''
    https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
    input:
        ray_tri: ([float, float, float],) * 5
    output:
        float
    '''
    v0v1 = nbv.sub(tri_v0, tri_v1)
    v0v2 = nbv.sub(tri_v0, tri_v2)
    pvec = nbv.cross(ray_direction, v0v2)

    det = nbv.dot(v0v1, pvec)

    if abs(det) < 0.000001:
        return np.NINF

    invDet = 1.0 / det

    tvec = nbv.sub(tri_v0, ray_origin)
    u = nbv.dot(tvec, pvec) * invDet

    if u < 0 or u > 1:
        return np.NINF

    qvec = nbv.cross(tvec, v0v1)
    v = nbv.dot(ray_direction, qvec) * invDet

    if v < 0 or u + v > 1:
        return np.NINF

    t = nbv.dot(v0v2, qvec) * invDet

    return t


@nb.njit(fastmath=True)
def ray_segment_set_intersect(ray_pnt, ray_dir, segs):
    intersects = np.zeros(shape=(segs.shape[0]))

    for i in range(segs.shape[0]):
        intersects[i] = ray_segment_intersect(ray_pnt, ray_dir, segs[i][0], segs[i][1])

    intersects = intersects[intersects >= 0.0]

    return intersects


@nb.njit(fastmath=True)
def ray_segment_intersect(ray_pnt, ray_dir, pnt1, pnt2):
    thres = 0.000001

    seg_dir = nbv.sub(pnt1, pnt2)
    ray_dir = nbv.div(nbv.norm(ray_dir), ray_dir)

    # check if ray origin lie on segment
    vec1 = nbv.sub(ray_pnt, pnt1)
    vec2 = nbv.sub(ray_pnt, pnt2)

    origin_on_segment = nbv.norm(nbv.cross(vec1, vec2)) < thres
    normal = nbv.cross(seg_dir, ray_dir)

    if origin_on_segment:
        if nbv.dot(vec1, vec2) < thres:
            return 0.0
        else:
            if nbv.norm(normal) < thres:
                dist1 = nbv.dot(vec1, ray_dir)
                dist2 = nbv.dot(vec2, ray_dir)
                if dist1 > thres and dist2 > thres:
                    if dist1 < dist2:
                        return nbv.norm(vec1)
                    else:
                        return nbv.norm(vec2)
            else:
                return np.NINF

    # check if ray and segment are parallel
    if nbv.norm(normal) < thres:
        return np.NINF

    # check if ray lie on one side of segment
    if nbv.dot(vec1, ray_dir) < 0 and nbv.dot(vec2, ray_dir) < 0:
        return np.NINF

    # check if segment lie on one side of ray
    if nbv.dot(np.cross(vec1, ray_dir), nbv.cross(vec2, ray_dir)) > 0:
        return np.NINF

    seg_normal = nbv.cross(normal, seg_dir)
    seg_normal = nbv.div(nbv.norm(seg_normal), seg_normal)
    dist = nbv.dot(vec1, seg_normal) / nbv.dot(ray_dir, seg_normal)

    try:
        assert not math.isnan(dist), 'dist is not a number'
    except:
        return np.NINF
    return dist


@nb.njit(fastmath=True)
def points_in_polygon(bnd_pnts, verts, closed=True, normal=None):
    num_v = verts.shape[0]
    assert num_v > 1

    if num_v == 2:
        assert normal is not None

    if num_v > 2:
        vec1 = nbv.sub(verts[0], verts[1])
        vec2 = nbv.sub(verts[1], verts[2])
        normal = nbv.cross(vec1, vec2)

    if closed:
        new_verts = np.zeros(shape=(verts.shape[0] + 1, verts.shape[1]), dtype=verts.dtype)
        new_verts[:verts.shape[0]] = verts
        new_verts[-1] = verts[0]
        verts = new_verts

    idx = np.array([point_in_polygon(pnt, verts, normal) for pnt in bnd_pnts])
    in_pnts = bnd_pnts[idx]

    return in_pnts


@nb.njit(fastmath=True)
def point_in_polygon(the_pnt, verts, normal=None):
    dists = np.array([dist_pnt_line(the_pnt, verts[i], verts[i + 1], normal) for i in range(len(verts) - 1)])
    summation = 0

    for dist in dists:
        if dist <= 0.000001:
            summation += 1

    if summation > 0:
        return False
    else:
        return True


@nb.njit(fastmath=True)
def point_in_polygon_face_numba(face_pnts, query_pnt):
    """ Finds if a point lies within the bounds of a polygon.

    A 3D convex polygon has many faces, a face has a face plane where the face lies in.
    A face plane has an outward normal vector, which directs to outside of the polygon.
    A point to face plane distance defines a geometry vector, if the distance vector has an opposite direction with
     the outward normal vector, then the point is in "inside half space" of the face plane, otherwise,
      it is in "outside half space" of the face plane.
    A point is determined to be inside of the 3D polygon if the point is in "inside half space"
     for all faces of the 3D convex polygon.

    https://www.eecs.umich.edu/courses/eecs380/HANDOUTS/PROJ2/InsidePoly.html
    https://www.codeproject.com/Articles/1065730/Point-Inside-Convex-Polygon-in-Cplusplus
    :param the_pnt:
    :param verts:
    :param closed:
    :param normal:
    :return:
    """
    u = nbv.sub(face_pnts[0], face_pnts[1])
    v = nbv.sub(face_pnts[0], face_pnts[2])
    normal = nbv.cross(u, v)
    D = -(normal[0] * face_pnts[0][0] + normal[1] * face_pnts[0][1] + normal[2] * face_pnts[0][2])
    dis = query_pnt[0] * normal[0] + query_pnt[1] * normal[1] + query_pnt[2] * normal[2] + D
    distance = dis / nbv.calc_l2_norm(normal)

    if distance > 0:
        return False
    else:
        return True


@nb.njit(fastmath=True)
def dist_pnt_line(query_pnt, pnt0, pnt1, normal):
    """Calculates the distance of a query point from a line.

    Let a line in three dimensions be specified by two points pnt0 & pnt1.

    :param query_pnt: Point from which the distance of the line is being measured.
    :param pnt0: First point of line.
    :param pnt1: Second point of line.
    :return: Distance of query point from line.
    """
    query_dir = nbv.sub(pnt0, query_pnt)
    line_dir = nbv.sub(pnt0, pnt1)
    perp_dir = nbv.cross(normal, line_dir)
    perp_dir = nbv.div(nbv.calc_l2_norm(perp_dir), perp_dir)

    return nbv.dot(query_dir, perp_dir)


@nb.njit(fastmath=True)
def dist_pnt_from_line_numba(query_pnt, pnt0, pnt1):
    """Calculates the distance of a query point from a line.

    Let a line in three dimensions be specified by two points pnt0 & pnt1.

    :param query_pnt: Point from which the distance of the line is being measured.
    :param pnt0: First point of line.
    :param pnt1: Second point of line.
    :return: Distance of query point from line.
    """
    query_dir = query_pnt - pnt0
    line_dir = pnt1 - pnt0
    cross_product = nbv.cross(query_dir, line_dir)
    A = nbv.calc_l2_norm(cross_product)
    line_norm = nbv.calc_l2_norm(line_dir)

    result = A / line_norm

    return result


@nb.njit(fastmath=True)
def dist_point_plane_numba(pnt, pl_pnt, pl_normal):
    p_dir = nbv.sub(pl_pnt, pnt)
    dist = nbv.dot(p_dir, pl_normal)

    return dist


@nb.njit(fastmath=True)
def outer_radius_triangle(pt1, pt2, pt3):
    a = nbv.calc_l2_norm(nbv.sub(pt2, pt1))
    b = nbv.calc_l2_norm(nbv.sub(pt3, pt2))
    c = nbv.calc_l2_norm(nbv.sub(pt1, pt3))
    p = (a + b + c) / 2
    return a * b * c / (4 * math.sqrt(p * (p - a) * (p - b) * (p - c)))
