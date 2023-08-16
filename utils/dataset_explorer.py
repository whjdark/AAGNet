import random
import pathlib
import argparse
import json
import hashlib

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from OCC.Display.WebGl import threejs_renderer
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Extend.TopologyUtils import TopologyExplorer

from occwl.solid import Solid
from occwl.face import Face
from occwl.graph import face_adjacency



feat_names = ['chamfer', 'through_hole', 'triangular_passage', 'rectangular_passage', '6sides_passage',
              'triangular_through_slot', 'rectangular_through_slot', 'circular_through_slot',
              'rectangular_through_step', '2sides_through_step', 'slanted_through_step', 'Oring', 'blind_hole',
              'triangular_pocket', 'rectangular_pocket', '6sides_pocket', 'circular_end_pocket',
              'rectangular_blind_slot', 'v_circular_end_blind_slot', 'h_circular_end_blind_slot',
              'triangular_blind_step', 'circular_blind_step', 'rectangular_blind_step', 'round', 'stock'
              ]

face_colors = [
                (0.9700192609404378, 0.9055119492048388, 0.1323910958497898), 
                (0.06660504960373947, 0.8303813089118807, 0.18731932715332889), 
                (0.10215758587633339, 0.44758647359931925, 0.19743749570413038), 
                (0.39618326204551335, 0.62480565418795, 0.49263998623974803), 
                (0.9563194150570774, 0.6863431793453533, 0.40198773505084073), 
                (0.7130311335430903, 0.5230673415079722, 0.360958551997956), 
                (0.9546937583877466, 0.6021401628064251, 0.10398061899932864), 
                (0.128418629621174, 0.38339751306229297, 0.19158928190370528), 
                (0.9608394495112227, 0.8562415399879139, 0.35996379127307776), 
                (0.8447461411950761, 0.6094638042385847, 0.6270924499592639), 
                (0.608161974268185, 0.14829199916733193, 0.8045844806839375), 
                (0.3911100021120745, 0.4512360980634469, 0.4243274963243149), 
                (0.14587592017360218, 0.022838821343438398, 0.15571507918186522), 
                (0.8096958445411236, 0.7164091463852411, 0.10006398944389583), 
                (0.17637293645693264, 0.1958775455478048, 0.817706000786001), 
                (0.44944192621774237, 0.738938573906961, 0.47097575885431253), 
                (0.4988884139971932, 0.12540630349619342, 0.05117859638958533), 
                (0.7141989735141261, 0.10619575782538193, 0.40160785621449757), 
                (0.8907191760896118, 0.32853909664596714, 0.5617643232088937), 
                (0.003188679730863675, 0.2513818008038544, 0.31507520557618907), 
                (0.04338783996955187, 0.5109066219752398, 0.01751921372339693), 
                (0.08918523237871268, 0.09105427694981261, 0.2694775316636171), 
                (0.6080768096407021, 0.34579812513326547, 0.8826508065977654), 
                (0.4926405898863041, 0.9728342822717221, 0.9958939931665864),
                (0.65, 0.65, 0.7)
              ]


def list_face(shape):
    '''
    input
        shape: TopoDS_Shape
    output
        fset: {TopoDS_Face}
    '''
    """
    fset = set()
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        s = exp.Current()
        exp.Next()
        face = topods.Face(s)
        fset.add(face)
    return list(fset)
    """
    topo = TopologyExplorer(shape)

    return list(topo.faces())


def load_body_from_step(step_file):
        """
        Load the body from the step file.  
        We expect only one body in each file
        """
        assert pathlib.Path(step_file).suffix in ['.step', '.stp', '.STEP', '.STP']
        reader = STEPControl_Reader()
        reader.ReadFile(str(step_file))
        reader.TransferRoots()
        shape = reader.OneShape()
        return shape


def load_json(pathname):
    with open(pathname, "r") as fp:
        return json.load(fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_file", type=str, required=True, help="Path to load the step file from")
    parser.add_argument("--label_file", type=str, required=True, help="Path to load the label file from")
    parser.add_argument("--visual_label", type=str, choices=['seg', 'inst', 'bottom'], required=True, help="which label to visualize")
    parser.add_argument("--show_stock", help='Whether to show stock faces in instance label visualization', action='store_true')
    parser.add_argument("--show_fag", help='Whether to show face adjacency graph', action='store_true')
    args = parser.parse_args()

    assert pathlib.Path(args.step_file).stem == pathlib.Path(args.label_file).stem
    shape = load_body_from_step(args.step_file)
    labels = load_json(args.label_file)
    
    # create graph
    faces_list = list_face(shape)
    num_faces = len(faces_list)

    step_name, sapmle_label = labels[0] 
    print(step_name)
    seg_label, inst_label, bottom_label = sapmle_label['seg'], sapmle_label['inst'], sapmle_label['bottom']

    # check map between face id and segmentaion label
    assert len(faces_list) == len(seg_label)
    print(seg_label)
    # check relation_matrix describing the feature instances
    assert len(faces_list) == len(inst_label)
    for row in inst_label:
        print(row)
    # check map between face id and botto identification label
    assert len(faces_list) == len(bottom_label)
    print(bottom_label)
    # instance adjacency numpy array to network graph
    relation_graph = nx.from_numpy_array(np.array(inst_label))

    # get face adjaceny graph (FAG)
    if args.show_fag:
        graph = face_adjacency(Solid(shape))
        face_adj = np.zeros((num_faces, num_faces))
        for face_idx in graph.nodes:
            # check the order of occwl.face_adjacency is equal to occ_utils.list_face
            assert graph.nodes[face_idx]["face"] == Face(faces_list[face_idx])
            # face_adj[face_idx][face_idx] = 1 # self-loop
            for neighbor in graph.neighbors(face_idx):
                face_adj[face_idx][neighbor] = 1
        # show FAG
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        fag = nx.from_numpy_array(face_adj)
        inst_adj_pos = nx.nx_pydot.graphviz_layout(relation_graph)
        # use same position layout to align fag with instance adjacency
        nx.draw_networkx(fag, pos=inst_adj_pos, with_labels=True, node_size=350)
        plt.show()

    # graph visualization
    if args.visual_label == 'seg':
        # visualize face segmantic label graph
        node_color = [None for _ in range(len(seg_label))] # default is background color
        for face_id in seg_label.keys():
            seg_lbl = seg_label[face_id]
            node_color[int(face_id)] = face_colors[seg_lbl]
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        nx.draw_networkx(relation_graph, pos=inst_adj_pos, with_labels=True, node_size=350, node_color=node_color, edge_color='w')
        plt.show()
    elif args.visual_label == 'inst':
        # visualize instance segmantic label graph
        node_color = [None for _ in range(len(seg_label))] # default is background color
        for face_id in seg_label.keys():
            seg_lbl = seg_label[face_id]
            node_color[int(face_id)] = face_colors[seg_lbl]
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        nx.draw_networkx(relation_graph, pos=inst_adj_pos, with_labels=True, node_size=350, node_color=node_color, edge_color='b')
        plt.show()
    elif args.visual_label == 'bottom':
        # visualize bottom face segmantic label graph
        node_color = [face_colors[24] for _ in range(len(seg_label))] # default is background color
        for face_id in seg_label.keys():
            if bottom_label[face_id]:
                node_color[int(face_id)] = (0,1,0)
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        nx.draw_networkx(relation_graph, pos=inst_adj_pos, with_labels=True, node_size=350, node_color=node_color, edge_color='w')
        plt.show()

    # 3D label visualization
    my_renderer = threejs_renderer.ThreejsRenderer()
    all_faces = list_face(shape)
    if args.visual_label == 'seg':
        # display face type label
        for face in all_faces:
            face_id = all_faces.index(face)
            face_type = seg_label[str(face_id)]
            face_color = face_colors[face_type]
            my_renderer.DisplayShape(face, color=face_color)
        my_renderer.render()
    elif args.visual_label == 'inst':
        # display instance label
        # display all faces
        if args.show_stock:
            for face in all_faces:
                my_renderer.DisplayShape(face)
        # display faces of each instance
        instances = set() # delete repeat
        for row_idx, row in enumerate(inst_label):
            if np.sum(row) <= 1e-6 : # stock face, no linked face
                continue
            # non-stock face
            instance = set() # delete repeat
            for col_idx, item in enumerate(row):
                if item: # have connections with currect face
                    instance.add(col_idx)
            instances.add(frozenset(instance)) # hashable set
        print('\n')
        print('This Shape Has {} instances'.format(len(instances)))
        print('\n')
        # draw instance with different random color
        for inst in instances:
            rnd_color = (random.random(), random.random(), random.random())
            for face_id in inst:
                my_renderer.DisplayShape(all_faces[face_id], color=rnd_color)
        my_renderer.render()
    elif args.visual_label == 'bottom':
        # display bottom face label
        # display each face
        for face in all_faces:
            face_id = all_faces.index(face)
            if bottom_label[str(face_id)]:
                my_renderer.DisplayShape(face, color=(0,1,0))
            else:
                my_renderer.DisplayShape(face)
        my_renderer.render()

    # !!!!!!!!!have problem, obsolete !!!!!!!!!!!!!!!
    # display, start_display, add_menu, add_function_to_menu = init_display()

    # ais_shp = AIS_ColoredShape(shape)
    # for face in bottom_map.keys():
    #     if bottom_map[face]:
    #         ais_shp.SetCustomColor(face, rgb_color(0,1,0))

    # display.Context.Display(ais_shp, True)
    # display.FitAll()

    # start_display()

    # OCC_DISPLAY, START_OCC_DISPLAY, ADD_MENU, _ = SimpleGui.init_display()
    # OCC_DISPLAY.EraseAll()

    # OCC_DISPLAY.DisplayShape(shape)
    # # OCC_DISPLAY = display_bounds(bounds, OCC_DISPLAY, color="blue")

    # OCC_DISPLAY.View_Iso()
    # OCC_DISPLAY.FitAll()

    # START_OCC_DISPLAY()

    # gui = BasicViewer()

    # # Explore the faces of the shape (these are known to be named)
    # exp = TopExp_Explorer(shape, TopAbs_FACE)
    # while exp.More():
    #     rgb = None
    #     s = exp.Current()
    #     exp.Next()
    #     item = tr.EntityFromShapeResult(s, 1)
    #     name = item.Name().ToCString()
    #     if name:
    #         print('Found entity: {}'.format(name))
    #         rgb = (1, 0, 0)
    #     gui.add(s, rgb)
