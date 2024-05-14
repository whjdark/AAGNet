import os
import glob
import json
from tqdm import tqdm
from occwl.solid import Solid
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.StepRepr import StepRepr_RepresentationItem
from occwl.solid import Solid
from occwl.graph import face_adjacency




def read_step_with_labels(filename):
    """
    Reads STEP file with labels on each B-Rep face.
    """
    if not os.path.exists(filename):
        print(filename, ' not exists')
        return

    reader = STEPControl_Reader()
    reader.ReadFile(filename)
    reader.TransferRoots()
    shape = reader.OneShape()

    treader = reader.WS().TransferReader()

    ids = []
    topo = TopologyExplorer(shape)
    faces = list(topo.faces())

    for face in faces:
        item = treader.EntityFromShapeResult(face, 1)
        if item is None:
            print(face)
            continue
        item = StepRepr_RepresentationItem.DownCast(item)
        name = item.Name().ToCString()
        if name:
            nameid = name
            # id_map[face] = nameid
            ids.append(int(nameid))

    return ids


def save_graph(graph, graph_path, shape_name):
    with open(os.path.join(graph_path, shape_name + '.json'), 'w', encoding='utf-8') as file:
        json.dump(graph, file)


def generate_graph(shape_dir, graph_path, shape_name):
    ids = read_step_with_labels(os.path.join(shape_dir, shape_name+'.step'))
    #print(id_map)
    save_graph(ids, graph_path, shape_name)
    return 1


if __name__ == '__main__':
    shape_dir = "steps"
    graph_dir = "labels"

    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)
    
    shape_paths = glob.glob(os.path.join(shape_dir, '*.step'))
    shape_names = [shape_path.split(os.sep)[-1].split('.')[0] for shape_path in shape_paths]

    for shape_name in tqdm(shape_names):
        # print(shape_name)
        generate_graph(shape_dir, graph_dir, shape_name)

    # use multi-process
    

