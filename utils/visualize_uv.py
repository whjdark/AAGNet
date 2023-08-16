import argparse
import numpy as np
from occwl.viewer import Viewer
from occwl.graph import face_adjacency
from occwl.uvgrid import ugrid, uvgrid
from occwl.io import load_step
from occwl.edge import Edge
from occwl.solid import Solid

import torch
import dgl
from dgl.data.utils import load_graphs



def build_graph(solid, curv_num_u_samples, surf_num_u_samples, surf_num_v_samples):
    # Build face adjacency graph with B-rep entities as node and edge features
    graph = face_adjacency(solid)

    # Compute the UV-grids for faces
    graph_face_feat = []
    for face_idx in graph.nodes:
        # Get the B-rep face
        face = graph.nodes[face_idx]["face"]
        # Compute UV-grids
        points = uvgrid(
            face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        normals = uvgrid(
            face, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        visibility_status = uvgrid(
            face, method="visibility_status", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        mask = np.logical_or(visibility_status == 0, visibility_status == 2)  # 0: Inside, 1: Outside, 2: On boundary
        # Concatenate channel-wise to form face feature tensor
        face_feat = np.concatenate((points, normals, mask), axis=-1)
        graph_face_feat.append(face_feat)
    graph_face_feat = np.asarray(graph_face_feat)

    # Compute the U-grids for edges
    graph_edge_feat = []
    for edge_idx in graph.edges:
        # Get the B-rep edge
        edge = graph.edges[edge_idx]["edge"]
        # Ignore dgenerate edges, e.g. at apex of cone
        if not edge.has_curve():
            continue
        # Compute U-grids
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        # Concatenate channel-wise to form edge feature tensor
        edge_feat = np.concatenate((points, tangents), axis=-1)
        graph_edge_feat.append(edge_feat)
    graph_edge_feat = np.asarray(graph_edge_feat)

    # Convert face-adj graph to DGL format
    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    dgl_graph = dgl.graph((src, dst), num_nodes=len(graph.nodes))
    dgl_graph.ndata["x"] = torch.from_numpy(graph_face_feat)
    dgl_graph.edata["x"] = torch.from_numpy(graph_edge_feat)
    return dgl_graph


def draw_face_uvgrids(solid, graph, viewer):
    a_face_data = graph.ndata["x"][14]
    face_uvgrids = a_face_data.view(-1, 7)
    points = []
    normals = []
    for idx in range(face_uvgrids.shape[0]):
        # Don't draw points outside trimming loop
        if face_uvgrids[idx, -1] == 0:
            continue
        points.append(face_uvgrids[idx, :3].cpu().numpy())
        normals.append(face_uvgrids[idx, 3:6].cpu().numpy())

    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)

    bbox = solid.box()
    max_length = max(bbox.x_length(), bbox.y_length(), bbox.z_length())

    # Draw the points
    viewer.display_points(
        points, color=(51.0 / 255.0, 0, 1), marker="point", scale=8*max_length
    )

    # Draw the normals
    # for pt, nor in zip(points, normals):
    #     viewer.display(Edge.make_line_from_points(pt, pt + nor * 0.1 * max_length), color=(51.0 / 255.0, 0, 1))


def draw_edge_uvgrids(solid, graph, viewer):
    a_face_data = graph.edata["x"][10]
    edge_uvgrids = a_face_data.view(-1, 6)
    points = []
    tangents = []
    for idx in range(edge_uvgrids.shape[0]):
        points.append(edge_uvgrids[idx, :3].cpu().numpy())
        tangents.append(edge_uvgrids[idx, 3:6].cpu().numpy())

    points = np.asarray(points, dtype=np.float32)
    tangents = np.asarray(tangents, dtype=np.float32)

    bbox = solid.box()
    max_length = max(bbox.x_length(), bbox.y_length(), bbox.z_length())

    # Draw the points
    viewer.display_points(points, color=(1, 0, 1), marker="point", scale=3*max_length)

    # Draw the tangents
    # for pt, tgt in zip(points, tangents):
    #    viewer.display(Edge.make_line_from_points(pt, pt + tgt * 0.1 * max_length), color=(1, 0, 1))


def draw_graph_edges(solid, graph, viewer):
    src, dst = graph.edges()
    num_u = graph.ndata["x"].shape[1]
    num_v = graph.ndata["x"].shape[2]
    bbox = solid.box()
    max_length = max(bbox.x_length(), bbox.y_length(), bbox.z_length())

    for s, d in zip(src, dst):
        src_pt = graph.ndata["x"][s, num_u // 2, num_v // 2, :3].cpu().numpy()
        dst_pt = graph.ndata["x"][d, num_u // 2, num_v // 2, :3].cpu().numpy()
        # Make a cylinder for each edge connecting a pair of faces
        up_dir = dst_pt - src_pt
        height = np.linalg.norm(up_dir)
        if height > 1e-3:
            v.display(
                Solid.make_cylinder(
                    radius=0.01 * max_length, height=height, base_point=src_pt, up_dir=up_dir
                ),
                color="BLACK",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Visualize UV-grids and face adj graphs for testing"
    )
    parser.add_argument("solid", type=str, help="Solid STEP file")
    args = parser.parse_args()

    solid = load_step(args.solid)[0]
    solid = solid.scale_to_unit_box()
    graph = build_graph(solid, 5, 5, 5)

    v = Viewer(backend="wx")
    # Draw the solid
    v.display(solid)
    # Draw the face UV-grids
    draw_face_uvgrids(solid, graph, viewer=v)
    # Draw the edge UV-grids
    # draw_edge_uvgrids(solid, graph, viewer=v)
    # Draw face-adj graph edges
    # draw_graph_edges(solid, graph, viewer=v)

    v.fit()
    v.show()
