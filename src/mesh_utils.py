from typing import List, Tuple, Set, Dict, Optional, TypeAlias, Iterator
from collections import defaultdict, Counter
from dataclasses import dataclass
from pprint import pprint
import numpy as np
import math
import glob
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import networkx as nx
import network_utils
import symmetry_compressor as symm
import pickle
from pathlib import Path
import graphlets
import argparse
import re


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
    """Compress a PLY file's connectivity data using graphlet atlas compression."""
    in_file = Path(input_file)
    if not in_file.exists():
        raise FileNotFoundError(f"File {in_file} not found!")
    
    orig_size = in_file.stat().st_size
    if verbose: print(f"original size: {orig_size} bytes")

    out_folder = in_file.parent / f"{in_file.stem}_AtlasCompressed"
    out_folder.mkdir(exist_ok=True)
    plydata = PlyData.read(input_file)
    if (elem_names := set(el.name for el in plydata.elements)) != {"vertex", "face"}:
        raise ValueError(F"Only vertex-face PLY files are supported, found {elem_names}")
    
    wireframe, face_degrees = skeleton(plydata)
    # wireframe = face_adjacency_graph(plydata)
    if len(face_degrees) == 1:
        face_degree = face_degrees[0]
    else:
        print(f"WARNING: mesh has faces of different degrees: {face_degrees} (decompression not supported!)")
        face_degree = None
    
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

    # compress the connectivity data
    # wf_comp = symm.compress_bipartite(wireframe, caching_mode=symm.CachingMode.DYNAMIC)
    wf_comp = graphlets.compress_subgraphlets(wireframe, **compressor_kwargs)
    if verbose: print(f"wireframe compressed to {wf_comp}")

    # serialize compressed graph
    edge_path = out_folder / f"{in_file.stem}_{face_degree}-gons.acgf"
    wf_comp.serialize(edge_path)
    graph_size = edge_path.stat().st_size
    if verbose: print(f"graph size: {graph_size} bytes")
    if verbose: print(f"total size: {vtxdata_sz + graph_size} bytes")
    
    print(f"output files in {out_folder}")
    if not ignored_props:
        conn_size = orig_size - vtxdata_sz
        conn_ratio = graph_size / conn_size
        print(f"connectivity zipped to {100 * conn_ratio:.1f}%")

        total_ratio = (vtxdata_sz + graph_size) / orig_size
        print(f"total size reduced to {100 * total_ratio:.1f}%")
        return 1 - total_ratio # relative efficiency
    else:
        return None


def cycles(graph: nx.Graph, length: int) -> Iterator[List[int]]:
    """Generate cycles of a given length in a graph."""
    # NOTE: can we just use nx.simple_cycles instead? chords on faces are very strange for a mesh
    # complexity of nx.simple_cycles: O((f + n)(len - 1)d^len) for f faces, mean degree d
    # for cycle in nx.chordless_cycles(graph, length_bound=length):
    for cycle in nx.simple_cycles(graph, length_bound=length):
        if len(cycle) == length:
            yield cycle    

def decompress_ply(folder: str, pbar=True, **kwargs) -> PlyData:
    folder: Path = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder} not found!")
    
    vtx_path: Path = next(folder.glob("*_vertices.ply")) # TODO: proper error reporting
    edge_path: Path = next(folder.glob("*.acgf"))

    # load vertex data
    plydata = PlyData.read(vtx_path)

    # load compressed wireframe
    wf_comp = graphlets.AtlasCompressedGraph.deserialize(edge_path)
    wireframe = wf_comp.decompress(in_place=True)

    # reconstruct face data
    if (mtch := re.match(r".*_(\d+)-gons", edge_path.stem)) is not None:
        face_degree = int(mtch.group(1))
    else:
        face_degree = 3
        print(f"WARNING: face degree not found in filename {edge_path.stem}, assuming triangular faces")

    faces = cycles(wireframe, length=face_degree)
    if pbar:
        f = len(wireframe.edges) - len(wireframe.nodes) + 2 # good estimate assuming (roughly) planar graph
        faces = tqdm(faces, total=f, desc="generating faces")

    faces = list(faces)
    # face_data = np.array([np.array(f) for f in faces], dtype=("vertex_indices", "u4", (face_degree,)))
    face_data = np.array([(f,) for f in faces],
                         dtype=[('vertex_indices', 'u4', (face_degree,))])
    # NOTE: is this the correct type? (serialization uses 4-byte unsigned int)
    plydata.elements = [plydata["vertex"], PlyElement.describe(face_data, "face")]

    out_file = folder / f"decompressed.ply"
    plydata.write(out_file)
    print(f"Decompressed mesh saved to {out_file}")
    return plydata


if __name__ == "__main__":
    # mesh_path = r"data\MeshLab_sample_meshes\non_manif_hole.ply"
    parser = argparse.ArgumentParser(description="(de-)compress PLY files using graphlet atlas compression.")
    subparsers = parser.add_subparsers(dest="command")
    
    zip_parser = subparsers.add_parser("zip", help="Compress a PLY file.")
    zip_parser.add_argument("input_file", type=str, help="Path to the input PLY file.")
    zip_parser.add_argument("--max_graphlet", type=int, default=5, help="Maximum graphlet size to compress.")
    zip_parser.add_argument("--verbose", action="store_true", help="Print more compression statistics.")

    unzip_parser = subparsers.add_parser("unzip", help="Decompress a PLY file.")
    unzip_parser.add_argument("folder", type=str, help="Path to the folder containing the compressed data.")
    
    args = parser.parse_args()
    if args.command == "zip":    
        compress_ply(args.input_file, max_graphlet_sz=args.max_graphlet, verbose=args.verbose, print_stats=args.verbose)
    elif args.command == "unzip":
        decompress_ply(args.folder)
