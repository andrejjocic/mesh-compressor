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
import pickle
import pathlib
import graphlets
from collections import Counter

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


def skeleton(data: PlyData, pbar=False) -> nx.Graph:
    """1-skeleton of the polygonal mesh, and list of unique face degrees."""
    G = nx.Graph(name="1-Skeleton")
    faces = data['face']['vertex_indices']
    degrees = Counter(map(len, faces)).keys()
    if len(degrees) > 1:
        print(f"NOTE: mesh has faces of different degrees: {degrees}")

    if pbar: faces = tqdm(faces, desc="traversing faces")
    
    for face in faces:
        for i in range(len(face)):
            edge = (face[i], face[(i + 1) % len(face)])
            G.add_edge(*edge)

    return G, list(degrees)

def face_adjacency_graph(data: PlyData) -> nx.Graph:
    """Construct face adjacency graph from a PLY file.
    NOTE: assuming face vertices describe facial walks! (doesn't matter for triangulations)"""
    edge2faces = defaultdict(list)

    for face in data['face']['vertex_indices']:
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


def assertEqualGraphs(g1: nx.Graph, g2: nx.Graph):
    assert g1.number_of_nodes() == g2.number_of_nodes()
    assert g1.number_of_edges() == g2.number_of_edges()
    # no need to complicate with isomorphism test, decompressed graph
    # should have the same nodes and edges (provided no isolated nodes?)
    assert set(g1.nodes) == set(g2.nodes)
    assert set(g1.edges) == set(g2.edges)


def compress_ply(input_file: str, verbose=False, **compressor_kwargs):
    """Compress a PLY file connectivity using graphlet atlas compression."""
    in_file = pathlib.Path(input_file)
    if not in_file.exists():
        raise FileNotFoundError(f"File {in_file} not found!")
    
    orig_size = in_file.stat().st_size
    if verbose: print(f"original size: {orig_size} bytes")

    out_folder = in_file.parent / f"{in_file.stem}_AtlasCompressed"
    out_folder.mkdir(exist_ok=True)
    plydata = PlyData.read(input_file)
    if set(el.name for el in plydata.elements) != {"vertex", "face"}:
        raise ValueError("Only vertex-face PLY files are supported!")
    
    wireframe, face_degrees = skeleton(plydata)
    if len(face_degrees) == 1:
        face_degree = face_degrees[0]
    else:
        print(f"WARNING: mesh has faces of different degrees: {face_degrees} (decompression not supported!)")
        face_degree = None
    # wireframe = face_adjacency_graph(plydata)
    # wf_comp = symm.compress_bipartite(wireframe, caching_mode=symm.CachingMode.DYNAMIC)
    wf_comp = graphlets.compress_subgraphlets(wireframe, **compressor_kwargs)

    print(f"wireframe compressed to {wf_comp}")
    
    # copy header and vertex data from input_file to output_file
    vtx_path = out_folder / f"{in_file.stem}_vertices.ply"

    ignored_props = [p.name for p in plydata["face"].properties if p.name != "vertex_indices"]
    for propname in ignored_props:
        # TODO: copy over properties of face element (so they can be matched to faces)
        # see https://app.todoist.com/app/task/support-face-data-6VJ966293G827VgF
        print(f"WARNING: ignoring property {propname} of face element! (compression ratio incorrect)")

    PlyData([plydata["vertex"]]).write(vtx_path)
    vtxdata_sz = vtx_path.stat().st_size
    if verbose: print(f"vertices size: {vtxdata_sz} bytes")

    # serialize compressed graph
    edge_path = out_folder / f"{in_file.stem}_{face_degree}-gons.acgf"
    wf_comp.serialize(edge_path)
    graph_size = edge_path.stat().st_size
    if verbose: print(f"graph size: {graph_size} bytes")
    if verbose: print(f"total size: {vtxdata_sz + graph_size} bytes")
    
    if not ignored_props:
        conn_size = orig_size - vtxdata_sz
        conn_effic = 1 - graph_size / conn_size
        print(f"connectivity efficiency ratio: {conn_effic:.3f}")

        rel_eff = 1 - (vtxdata_sz + graph_size) / orig_size
        print(f"total efficiency ratio: {rel_eff:.3f}")
        return rel_eff
    else:
        return None
    

def decompress_ply():
    raise NotImplementedError("Decompression not yet implemented!")


if __name__ == "__main__":
    mesh_path = r"data\MeshLab_sample_meshes\non_manif_hole.ply"
    # mesh_path = r"data\MeshLab_sample_meshes\bunny10k.ply"
    compress_ply(mesh_path, max_graphlet_sz=6)
    # S = skeleton(data)
    # print(S)
    # Sc_stat = symm.compress_bipartite(S, caching_mode=symm.CachingMode.STATIC)
    # print(Sc_stat)
    # assertEqualGraphs(S, Sc_stat.decompr  ss())
    # print("-"*80)
    # Scomp = symm.compress_bipartite(S, caching_mode=symm.CachingMode.DYNAMIC)
    # print(Scomp)
    # assertEqualGraphs(S, Scomp.decompress())

    # F = face_adjacency_graph(data)
    # network_utils.info(F)
    # Fcomp = symm.compress_bipartite(F, caching_mode=symm.CachingMode.DYNAMIC)
    # print(Fcomp)
    # assertEqualGraphs(F, Fcomp.decompress())