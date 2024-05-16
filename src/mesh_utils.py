from typing import List, Tuple, Set, Dict, Optional, TypeAlias, Iterator
from collections import defaultdict
from dataclasses import dataclass
from pprint import pprint
import gudhi
from gudhi.subsampling import sparsify_point_set
import numpy as np
import math
import glob
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import networkx as nx
import network_utils
import symmetry_compressor as symm


def export_ply(triangulation, points: List[Tuple[float, float, float]], description=""):
    """Export a triangulation to a PLY file."""
    triangulation = [t for t in triangulation
                     if len(t) == 3]
    header = f"""ply
format ascii 1.0           
comment Exported triangulation
comment {description}
element vertex {len(points)}
property float x
property float y
property float z
element face {len(triangulation)}
property list uchar int vertex_index
end_header
"""
    data = [header] 
    for point in points:
        data.append(" ".join(map(str, point)) + "\n")
    for simplex in triangulation:
        line = "{} ".format(len(simplex))
        line += " ".join(map(str, simplex))
        line += "\n"
        data.append(line)

    return "".join(data)


def skeleton(data: PlyData) -> nx.Graph:
    """1-skeleton of the polygonal mesh."""
    G = nx.Graph(name="1-Skeleton")
    for face in tqdm(data['face']['vertex_indices'], desc="traversing faces"):
        for i in range(len(face)):
            edge = (face[i], face[(i + 1) % len(face)])
            G.add_edge(*edge)

    return G

def face_adjacency_graph(data: PlyData) -> nx.Graph:
    """Construct face adjacency graph from a PLY file.
    NOTE: assuming face vertices describe facial walks! (doesn't matter for triangulations)"""
    edge2faces = defaultdict(list)

    for face in tqdm(data['face']['vertex_indices'], desc="traversing faces"):
        face = tuple(face)
        for i in range(len(face)):
            edge = (face[i], face[(i + 1) % len(face)])
            if edge[0] > edge[1]:
                edge = (edge[1], edge[0]) # may appear in different order for different faces

            edge2faces[edge].append(face)

    G = nx.Graph(name="Face adjacency graph")
    for face_list in edge2faces.values():
        if len(face_list) == 2:
            G.add_edge(*face_list)
        elif len(face_list) > 2:
            raise ValueError("Non-manifold mesh detected!")
        
    return G
    

if __name__ == "__main__":
    data = PlyData.read(r"data\MeshLab_sample_meshes\non_manif_hole.ply")

    S = skeleton(data)
    print(S)
    print(type((next(iter(S.nodes)))))
    Scomp = symm.compress_bipartite(S)
    print(Scomp)
    # F = face_adjacency_graph(data)
    # network_utils.info(F)