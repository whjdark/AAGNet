
##Copyright 2008-2013 Jelle Feringa (jelleferinga@gmail.com)
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

from OCC.Core.TopoDS import TopoDS_Solid

from OCCUtils.Topology import Topo
from OCCUtils.base import GlobalProperties, BaseObject
from OCCUtils.shell import Shell


class Solid(TopoDS_Solid, BaseObject):
    def __init__(self, solid):
        assert isinstance(solid, TopoDS_Solid), 'need a TopoDS_Solid, got a %s' % solid.__class__
        assert not solid.IsNull()
        super(Solid, self).__init__()
        BaseObject.__init__(self, 'solid')
        # we need to copy the base shape using the following three
        # lines
        assert self.IsNull()
        self.TShape(solid.TShape())
        self.Location(solid.Location())
        self.Orientation(solid.Orientation())
        assert not self.IsNull()

        self.GlobalProperties = GlobalProperties(self)

    def shells(self):
        return (Shell(sh) for sh in Topo(self))