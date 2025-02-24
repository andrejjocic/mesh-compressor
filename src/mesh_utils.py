from typing import List, Tuple, Set, Dict, Optional, TypeAlias, Iterator, Any
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import networkx as nx
import network_utils
import symmetry_compressor as symm
from pathlib import Path
import graphlets
import argparse
import re
from orientable import orient, Vertex, Edge, Triangle, Simplex, SimplexMap
from timeit import default_timer as time
import struct


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


def compress_ply(input_file: str, encode_holes=True, verbose=False, **compressor_kwargs):
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
        print(f"WARNING: ignoring property {propname} of face element! (compression ratio incorrect)")

    PlyData([plydata["vertex"]]).write(vtx_path)
    vtxdata_sz = vtx_path.stat().st_size
    if verbose: print(f"vertices size: {vtxdata_sz} bytes")

    # compress the connectivity data
    wf_comp = graphlets.compress_subgraphlets(wireframe, **compressor_kwargs)
    print(f"wireframe compressed to {wf_comp}")

    # serialize compressed graph
    conn_path = out_folder / f"{in_file.stem}_{face_degree}-gons.acgf"
    wf_comp.serialize(conn_path)

    if encode_holes:
        if verbose: print("checking for holes in the mesh...")
        actual_faces = {tuple(sorted(face)) for face in plydata["face"]["vertex_indices"]}
        holes = []
        for potential_face in surface_faces(wireframe, face_degree, pbar=False):
            if tuple(sorted(potential_face)) not in actual_faces:
                holes.append(potential_face)
        
        if verbose: print(f"found {len(holes)} holes in the mesh")

        # append them to the compressed connectivity data
        with open(conn_path, "ab") as f:
            for hole in holes:
                f.write(struct.pack("<I", len(hole))) # NOTE: not needed if only supporting regular meshes
                f.write(struct.pack(f"<{len(hole)}I", *hole))


    graph_size = conn_path.stat().st_size
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


def cycles(graph: nx.Graph, length: int, ensure_chordless: bool) -> Iterator[List[int]]:
    """Generate cycles of a given length in a graph.
    
    If ensure_chordless is True, explicitly check that output cycles are chordless
    (i.e. no edges between non-adjacent vertices). Note that this should not happen for regular
    meshes (with faces of the same degree).

    Otherwise the complexity is O((f + n)(len - 1)d^len) for f faces, mean degree d
    """
    cycle_gen = nx.chordless_cycles if ensure_chordless else nx.simple_cycles
    for cycle in cycle_gen(graph, length_bound=length):
        if len(cycle) == length:
            yield cycle    


def triangles(graph: nx.Graph) -> Iterator[List[Any]]:
    """Generate all unique triangles in a graph."""
    G = nx.convert_node_labels_to_integers(graph, first_label=0, label_attribute="old_label")
    n = G.number_of_nodes()

    def common_neighbors(u, v):
        return set(G[u]) & set(G[v]) # OPT: linear time with sorted adjacency lists

    for u in range(n - 2):
        for v in G[u]:
            if u < v:
                for w in common_neighbors(u, v):
                    if v < w:
                        yield [G.nodes[x]["old_label"] for x in (u, v, w)]


def surface_faces(graph: nx.Graph, face_degree: int, pbar: bool) -> List[List[int]]:
    """Compute the facial walks (of length `face_degree`) of a graph.
    Filter out all the cycles under or above the surface (ones with all edges incident on more than 2 faces)."""

    all_cycles = triangles(graph) if face_degree == 3 else cycles(graph, length=face_degree)
    if pbar:
        f = len(graph.edges) - len(graph.nodes) + 2 # good estimate assuming (roughly) planar graph
        all_cycles = tqdm(all_cycles, total=f, desc="generating faces", ncols=100)

    all_cycles = list(all_cycles)

    cycles_on_edge = SimplexMap(key_dimension=1, map_constructor=lambda: defaultdict(list))
    for cycle in all_cycles:
        for edge in zip(cycle, cycle[1:] + cycle[:1]):
            cycles_on_edge[edge].append(cycle)

    surface = []
    boundary_size = 0

    for cycle in all_cycles:
        edges = list(zip(cycle, cycle[1:] + cycle[:1]))
        nonmanifold_edges = list(filter(lambda e: len(cycles_on_edge[e]) > 2, edges))
        boundary_edges = list(filter(lambda e: len(cycles_on_edge[e]) == 1, edges))
        boundary_size += len(boundary_edges)
        internal_edges = len(cycle) - len(boundary_edges)

        if len(boundary_edges) == len(edges):
            print(f"WARNING: face {cycle} is isolated")

        if (n_nonmf := len(nonmanifold_edges)) != internal_edges:
            surface.append(cycle)
        # if 1 <= n_nonmf < internal_edges, this just means the face is *touching* an invalid face
        if n_nonmf > internal_edges:
            # this case impossible if we computed the SimplexMap correctly (?)
            print(f"WARNING: face {cycle} has {internal_edges} internal and {n_nonmf} non-manifold edges")

    print(f"removed {len(all_cycles) - len(surface)} non-surface faces") # TODO: verbose flag check
    print(f"found {boundary_size} boundary edges")
    return surface            
    

def decompress_ply(folder: str, out_name: str, pbar=False, set_orientation=True, flip_faces=False) -> PlyData:
    """Decompress a PLY file's connectivity data using graphlet atlas compression.
    ### Parameters:
    - folder: path to the folder containing the compressed data
    - pbar: show progress bar
    - encode_holes: check for single-face holes in the mesh so they may be preserved in the decompressed mesh
    - set_orientation: orient the faces of the decompressed mesh (if they form an orientable 2-manifold)
    - flip_faces: flip the orientation of all faces, w.r.t. the (arbitrary) default orientation"""
    folder: Path = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder} not found!")
    
    vtx_path: Path = next(folder.glob("*_vertices.ply")) # TODO: proper error reporting
    conn_path: Path = next(folder.glob("*.acgf"))

    # load vertex data
    plydata = PlyData.read(vtx_path)

    # load compressed wireframe
    wf_comp, bytes_read = graphlets.AtlasCompressedGraph.deserialize(conn_path)

    if (mtch := re.match(r".*_(\d+)-gons", conn_path.stem)) is not None:
        face_degree = int(mtch.group(1))
    else:
        face_degree = 3
        print(f"WARNING: face degree not found in filename {conn_path.stem}, assuming triangular faces")
    
    # check for holes in the mesh
    if bytes_read < conn_path.stat().st_size:
        SIZEOF_INT = 4 # bytes
        with open(conn_path, "rb") as f:
            f.seek(bytes_read)
            holes = set()
            while f.tell() < conn_path.stat().st_size:
                hole_sz = struct.unpack("<I", f.read(SIZEOF_INT))[0]
                hole = struct.unpack(f"<{hole_sz}I", f.read(SIZEOF_INT * hole_sz))
                holes.add(tuple(sorted(hole)))
        
        print(f"recovered {len(holes)} holes from the original mesh")
    else:
        holes = None

    wireframe = wf_comp.decompress(in_place=True)
    faces = surface_faces(wireframe, face_degree, pbar)

    # remove holes from faces (NOTE: should we do this before or after setting orientation?)
    if holes is not None:
        nf = len(faces)
        faces = [f for f in faces if tuple(sorted(f)) not in holes]
        print(f"removed {nf - len(faces)} faces (holes)")

    if set_orientation and face_degree != 3:
        print(f"WARNING: face orientation not supported for face degree {face_degree}")
    if set_orientation and face_degree == 3: # TODO: support other face degrees
        try:
            ori_faces = orient([tuple(face) for face in faces], reverse_first=flip_faces)
            if ori_faces is not None:
                faces = [list(f) for f in ori_faces] # can we just keep them as tuples?
            else:
                print("Warning: mesh is not orientable, face orientation not set!")
        except Exception as e:
            print(f"Error setting face orientation: {e}")

    # face_data = np.array([np.array(f) for f in faces], dtype=("vertex_indices", "u4", (face_degree,)))
    face_data = np.array([(f,) for f in faces], dtype=[('vertex_indices', 'u4', (face_degree,))])
    # NOTE: is this the correct type? (serialization uses 4-byte unsigned int)
    plydata.elements = [plydata["vertex"], PlyElement.describe(face_data, "face")]

    plydata.write(out_file := folder / f"{out_name}.ply")
    print(f"Decompressed mesh saved to {out_file}")
    return plydata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="(de-)compress PLY files using graphlet atlas compression.")
    parser.add_argument("--time", action="store_true", help="Print execution time.")
    subparsers = parser.add_subparsers(dest="command")
    
    zip_parser = subparsers.add_parser("zip", help="Compress a PLY file.")
    zip_parser.add_argument("input_file", type=str, help="Path to the input PLY file.")
    zip_parser.add_argument("--max_graphlet", type=int, default=5, help="Maximum graphlet size to search for (default=5, max=7).")
    zip_parser.add_argument("--verbose", action="store_true", help="Print more compression statistics.")
    zip_parser.add_argument("--preserve_holes", action="store_true",
                            help="Check for single-face holes in the mesh so they may be preserved (may take a few extra seconds).")
    # TODO: output path option

    unzip_parser = subparsers.add_parser("unzip", help="Decompress a PLY file.")
    unzip_parser.add_argument("folder", type=str, help="Path to the folder containing the compressed data.")
    unzip_parser.add_argument("--output_file", type=str, default="decompressed", help="Name of the output PLY file.")
    unzip_parser.add_argument("--pbar", action="store_true", help="Show (rough) progress bar for face enumeraton.")
    unzip_parser.add_argument("--no_orientation", action="store_false", help="Don't set face orientation.")
    unzip_parser.add_argument("--flip_orientation", action="store_true",
                              help="Flip the orientation of all faces, w.r.t. the (arbitrary) default orientation.")
    
    args = parser.parse_args()
    t0 = time()
    if args.command == "zip":    
        compress_ply(args.input_file, max_graphlet_sz=args.max_graphlet, encode_holes=args.preserve_holes,
                     verbose=args.verbose, print_stats=args.verbose)
    elif args.command == "unzip":
        decompress_ply(args.folder, out_name=args.output_file, pbar=args.pbar,
                       set_orientation=args.no_orientation, flip_faces=args.flip_orientation)

    if args.time:
        print(f"Execution time: {time() - t0:.6f} seconds")