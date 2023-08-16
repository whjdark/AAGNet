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

from OCC.Core.TopoDS import TopoDS_Shell
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Shell

from OCCUtils.Topology import Topo
from OCCUtils.base import BaseObject, GlobalProperties


class Shell(TopoDS_Shell, BaseObject):
    _n = 0

    def __init__(self, shell):
        assert isinstance(shell, TopoDS_Shell), 'need a TopoDS_Shell, got a %s' % shell.__class__
        assert not shell.IsNull()
        super(Shell, self).__init__()
        BaseObject.__init__(self, 'shell')
        # we need to copy the base shape using the following three
        # lines
        assert self.IsNull()
        self.TShape(shell.TShape())
        self.Location(shell.Location())
        self.Orientation(shell.Orientation())
        assert not self.IsNull()

        self.GlobalProperties = GlobalProperties(self)
        self._n += 1

    def analyse(self):
        """
        :return:
        """
        ss = ShapeAnalysis_Shell(self)
        if ss.HasFreeEdges():
            bad_edges = [e for e in Topo(ss.BadEdges()).edges()]
        return bad_edges

    def Faces(self):
        """
        :return:
        """
        return Topo(self, True).faces()

    def Wires(self):
        """
        :return:
        """
        return Topo(self, True).wires()

    def Edges(self):
        return Topo(self, True).edges()