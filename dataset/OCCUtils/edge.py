##Copyright 2008-2015 Jelle Feringa (jelleferinga@gmail.com)
##
##This file is part of pythonOCC.
##
##pythonOCC is free software: you can redistribute it and/or modify
##it under the terms of the GNU Lesser General Public License as published by
##the Free Software Foundation, either version 3 of the License, or
##(at your option) any later version.
##
##pythonOCC is distributed in the hope that it will be useful,
##but WITHOUT ANY WARRANTY; without even the implied warranty of
##MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##GNU Lesser General Public License for more details.
##
##You should have received a copy of the GNU Lesser General Public License
##along with pythonOCC.  If not, see <http://www.gnu.org/licenses/>

from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_HCurve
from OCC.Core.GCPnts import GCPnts_UniformAbscissa
from OCC.Core.Geom import Geom_OffsetCurve, Geom_TrimmedCurve
from OCC.Core.TopExp import topexp
from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Vertex, TopoDS_Face
from OCC.Core.gp import gp_Vec, gp_Dir, gp_Pnt
from OCC.Core.GeomLProp import GeomLProp_CurveTool
from OCC.Core.BRepLProp import BRepLProp_CLProps
from OCC.Core.GeomLib import geomlib
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Edge
from OCC.Core.BRep import BRep_Tool, BRep_Tool_Continuity
from OCC.Core.BRepIntCurveSurface import BRepIntCurveSurface_Inter

# high-level
from OCCUtils.Common import vertex2pnt, minimum_distance, assert_isdone, fix_continuity
from OCCUtils.Construct import make_edge
from OCCUtils.types_lut import geom_lut
from OCCUtils.base import BaseObject


class IntersectCurve(object):
    def __init__(self, instance):
        self.instance = instance

    def intersect(self, other, tolerance=1e-2):
        '''Intersect self with a point, curve, edge, face, solid
        method wraps dealing with the various topologies
        '''
        if isinstance(other, TopoDS_Face):
            face_curve_intersect = BRepIntCurveSurface_Inter()
            face_curve_intersect.Init(other, self.instance.adaptor.Curve(), tolerance)
            pnts = []
            while face_curve_intersect.More():
                next(face_curve_intersect)
                pnts.append(face_curve_intersect.Pnt())
            return pnts


class DiffGeomCurve(object):
    def __init__(self, instance):
        self.instance = instance
        self._local_props = BRepLProp_CLProps(self.instance.adaptor, 2, self.instance.tolerance)

    @property
    def _curvature(self):
        return self._local_props

    def radius(self, u):
        '''returns the radius at u
        '''
        # NOT SO SURE IF THIS IS THE SAME THING!!!
        self._curvature.SetParameter(u)
        pnt = gp_Pnt()
        self._curvature.CentreOfCurvature(pnt)
        return pnt

    def curvature(self, u):
        # ugly
        self._curvature.SetParameter(u)
        return self._curvature.Curvature()

    def tangent(self, u):
        '''sets or gets ( iff vector ) the tangency at the u parameter
        tangency can be constrained so when setting the tangency,
        you're constrainting it in fact
        '''
        self._curvature.SetParameter(u)
        if self._curvature.IsTangentDefined():
            ddd = gp_Dir()
            self._curvature.Tangent(ddd)
            return ddd
        else:
            raise ValueError('no tangent defined')

    def normal(self, u):
        '''returns the normal at u
        computes the main normal if no normal is found
        see:
        www.opencascade.org/org/forum/thread_645+&cd=10&hl=nl&ct=clnk&gl=nl
        '''
        try:
            self._curvature.SetParameter(u)
            a_dir = gp_Dir()
            self._curvature.Normal(a_dir)
            return a_dir
        except:
            raise ValueError('no normal was found')

    def derivative(self, u, n):
        '''
        returns n derivatives at parameter b
        '''
        self._curvature.SetParameter(u)
        deriv = {1: self._curvature.D1,
                 2: self._curvature.D2,
                 3: self._curvature.D3,
                 }
        try:
            return deriv[n]
        except KeyError:
            raise AssertionError('n of derivative is one of [1,2,3]')

    def points_from_tangential_deflection(self):
        pass

#===========================================================================
#    Curve.Construct
#===========================================================================


class ConstructFromCurve():
    def __init__(self, instance):
        self.instance = instance

    def make_offset(self, offset, vec):
        '''
        returns an offsetted curve
        @param offset: the distance between self.crv and the curve to offset
        @param vec:    offset direction
        '''
        return Geom_OffsetCurve(self.instance.h_crv, offset, vec)


class Edge(TopoDS_Edge, BaseObject):
    def __init__(self, edge):
        assert isinstance(edge, TopoDS_Edge), 'need a TopoDS_Edge, got a %s' % edge.__class__
        assert not edge.IsNull()
        super(Edge, self).__init__()
        BaseObject.__init__(self, 'edge')
        # we need to copy the base shape using the following three
        # lines
        assert self.IsNull()
        self.TShape(edge.TShape())
        self.Location(edge.Location())
        self.Orientation(edge.Orientation())
        assert not self.IsNull()

        # tracking state
        self._local_properties_init = False
        self._curvature_init = False
        self._geometry_lookup_init = False
        self._curve = None
        self._adaptor = None

        # instantiating cooperative classes
        # cooperative classes are distinct through CamelCaps from
        # normal method -> pep8
        self.DiffGeom = DiffGeomCurve(self)
        self.Intersect = IntersectCurve(self)
        self.Construct = ConstructFromCurve(self)

        # GeomLProp object
        self._curvature = None

    def is_closed(self):
        return self.adaptor.IsClosed()

    def is_periodic(self):
        return self.adaptor.IsPeriodic()

    def is_rational(self):
        return self.adaptor.IsRational()

    def continuity(self):
        return self.adaptor.Continuity

    def degree(self):
        if 'line' in self.type:
            return 1
        elif 'curve' in self.type:
            return self.adaptor.Degree()
        else:
            # hyperbola, parabola, circle
            return 2

    def nb_knots(self):
        return self.adaptor.NbKnots()

    def nb_poles(self):
        return self.adaptor.NbPoles()

    @property
    def curve(self):
        if self._curve is not None and not self.is_dirty:
            pass
        else:
            self._curve =  BRep_Tool().Curve(self)[0]
        return self._curve

    @property
    def adaptor(self):
        if self._adaptor is not None and not self.is_dirty:
            pass
        else:
            self._adaptor = BRepAdaptor_Curve(self)
        return self._adaptor


    @property
    def type(self):
        return geom_lut[self.adaptor.Curve().GetType()]

    def pcurve(self, face):
        """
        computes the 2d parametric spline that lies on the surface of the face
        :return: Geom2d_Curve, u, v
        """
        crv, u, v = BRep_Tool().CurveOnSurface(self, face)
        return crv, u, v

    def _local_properties(self):
        self._lprops_curve_tool = GeomLProp_CurveTool()
        self._local_properties_init = True

    def domain(self):
        '''returns the u,v domain of the curve'''
        return self.adaptor.FirstParameter(), self.adaptor.LastParameter()

#===========================================================================
#    Curve.GlobalProperties
#===========================================================================

    def length(self, lbound=None, ubound=None, tolerance=1e-5):
        '''returns the curve length
        if either lbound | ubound | both are given, than the length
        of the curve will be measured over that interval
        '''
        _min, _max = self.domain()
        if _min < self.adaptor.FirstParameter():
            raise ValueError('the lbound argument is lower than the first parameter of the curve: %s ' % (self.adaptor.FirstParameter()))
        if _max > self.adaptor.LastParameter():
            raise ValueError('the ubound argument is greater than the last parameter of the curve: %s ' % (self.adaptor.LastParameter()))

        lbound = _min if lbound is None else lbound
        ubound = _max if ubound is None else ubound
        return GCPnts_AbscissaPoint().Length(self.adaptor, lbound, ubound, tolerance)

#===========================================================================
#    Curve.modify
#===========================================================================

    def trim(self, lbound, ubound):
        '''
        trim the curve
        @param lbound:
        @param ubound:
        '''
        a, b = sorted([lbound, ubound])
        tr = Geom_TrimmedCurve(self.adaptor.Curve().Curve(), a, b)
        return Edge(make_edge(tr))

    def extend_by_point(self, pnt, degree=3, beginning=True):
        '''extends the curve to point
        does not extend if the degree of self.curve > 3
        @param pnt:
        @param degree:
        @param beginning:
        '''
        if self.degree > 3:
            raise ValueError('to extend you self.curve should be <= 3, is %s' % (self.degree))
        return geomlib.ExtendCurveToPoint(self.curve, pnt, degree, beginning)

#===========================================================================
#    Curve.
#===========================================================================
    def closest(self, other):
        return minimum_distance(self, other)

    def project_vertex(self, pnt_or_vertex):
        ''' returns the closest orthogonal project on `pnt` on edge
        '''
        if isinstance(pnt_or_vertex, TopoDS_Vertex):
            pnt_or_vertex = vertex2pnt(pnt_or_vertex)

        poc = GeomAPI_ProjectPointOnCurve(pnt_or_vertex, self.curve)
        return poc.LowerDistanceParameter(), poc.NearestPoint()

    def distance_on_curve(self, distance, close_parameter, estimate_parameter):
        '''returns the parameter if there is a parameter
        on the curve with a distance length from u
        raises OutOfBoundary if no such parameter exists
        '''
        gcpa = GCPnts_AbscissaPoint(self.adaptor, distance, close_parameter, estimate_parameter, 1e-5)
        with assert_isdone(gcpa, 'couldnt compute distance on curve'):
            return gcpa.Parameter()

    def mid_point(self):
        """
        :return: the parameter at the mid point of the curve, and
        its corresponding gp_Pnt
        """
        _min, _max = self.domain()
        _mid = (_min+_max) / 2.
        return _mid, self.adaptor.Value(_mid)

    def divide_by_number_of_points(self, n_pts, lbound=None, ubound=None):
        '''returns a nested list of parameters and points on the edge
        at the requested interval [(param, gp_Pnt),...]
        '''
        _lbound, _ubound = self.domain()
        if lbound:
            _lbound = lbound
        elif ubound:
            _ubound = ubound

        # minimally two points or a Standard_ConstructionError is raised
        if n_pts <= 1:
            n_pts = 2

        try:
            npts = GCPnts_UniformAbscissa(self.adaptor, n_pts, _lbound, _ubound)
        except:
            print("Warning : GCPnts_UniformAbscissa failed")
        if npts.IsDone():
            tmp = []
            for i in range(1, npts.NbPoints()+1):
                param = npts.Parameter(i)
                pnt = self.adaptor.Value(param)
                tmp.append((param, pnt))
            return tmp
        else:
            return None

    def __eq__(self, other):
        if hasattr(other, 'topo'):
            return self.IsEqual(other)
        else:
            return self.IsEqual(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def first_vertex(self):
        return topexp.FirstVertex(self)

    def last_vertex(self):
        return topexp.LastVertex(self)

    def common_vertex(self, edge):
        vert = TopoDS_Vertex()
        if topexp.CommonVertex(self, edge, vert):
            return vert
        else:
            return False

    def as_vec(self):
        if self.is_line():
            first, last = map(vertex2pnt, [self.first_vertex(), self.last_vertex()])
            return gp_Vec(first, last)
        else:
            raise ValueError("edge is not a line, hence no meaningful vector can be returned")

#===========================================================================
#    Curve.
#===========================================================================

    def parameter_to_point(self, u):
        '''returns the coordinate at parameter u
        '''
        return self.adaptor.Value(u)

    def fix_continuity(self, continuity):
        """
        splits an edge to achieve a level of continuity
        :param continuity: GeomAbs_C*
        """
        return fix_continuity(self, continuity)

    def continuity_from_faces(self, f1, f2):
        return BRep_Tool_Continuity(self, f1, f2)

#===========================================================================
#    Curve.
#===========================================================================

    def is_line(self):
        '''checks if the curve is planar
        '''
        if self.nb_knots() == 2 and self.nb_poles() == 2:
            return True
        else:
            return False

    def is_seam(self, face):
        """
        :return: True if the edge has two pcurves on one surface
        ( in the case of a sphere for example... )
        """
        sae = ShapeAnalysis_Edge()
        return sae.IsSeam(self, face)

    def is_edge_on_face(self, face):
        '''checks whether curve lies on a surface or a face
        '''
        return ShapeAnalysis_Edge().HasPCurve(self, face)

#===========================================================================
#    Curve.graphic
#===========================================================================
    def show(self):
        '''
        poles, knots, should render all slightly different.
        here's how...
        http://www.opencascade.org/org/forum/thread_1125/
        '''
        super(Edge, self).show()


if __name__ == '__main__':
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCCUtils.Topology import Topo
    b = BRepPrimAPI_MakeBox(10, 20, 30).Shape()
    t = Topo(b)
    ed = next(t.edges())
    my_e = Edge(ed)
    print(my_e.tolerance)