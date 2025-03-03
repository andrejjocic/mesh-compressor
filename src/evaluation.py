from typing import *
from pathlib import Path
from mesh_utils import skeleton
from plyfile import PlyData, PlyElement
import numpy as np
from symmetry_compressor import compress_bipartite, CachingMode
from collections import defaultdict
from matplotlib import pyplot as plt

def compression_ratio(original_size, compressed_size):
    return original_size / compressed_size

# def weissman_score()

def print_plyfiles(folder: str):
    folder = 'data/MeshLab_sample_meshes'
    plyfiles = Path(folder).rglob('*.ply')
    sorted_plyfiles = sorted(plyfiles, key=lambda x: x.stat().st_size)
    for plyfile in sorted_plyfiles:
        if "compressed" in str(plyfile).lower():
            continue
        plydata = PlyData.read(plyfile)
        try:
            face_degres = Counter(map(len, plydata['face'].data['vertex_indices']))
            tail = "[!!!]" if set(face_degres.keys()) != {3} or "bunny" in str(plyfile).lower() else ""
            print(f"{plyfile.stat().st_size:<10} {plyfile.stem} {face_degres} {tail}")
        except Exception as e:
            print(f"Error reading {plyfile.stem}: {e}") # triangle strip encoding?
            continue


def relative_conn_size(mesh: PlyData, conn_size_bytes: int) -> tuple[float, float]:
    """return (bpv, bpf)"""
    size_bits = conn_size_bytes * 8
    n_verts = num_verts(mesh)
    n_faces = num_faces(mesh)

    bpv = size_bits / n_verts
    bpf = size_bits / n_faces
    return bpv, bpf

def write_connectivity(mesh: PlyData, output_path: str, binary: bool = False) -> None:
    """Write face connectivity data to a PLY file"""
    face_list = [(face.tolist(),) for face in mesh['face']['vertex_indices']]
    faces_element = PlyElement.describe(
        np.array(face_list, dtype=[('vertex_indices', 'i4', (3,))]), 
        'face'
    )
    PlyData([faces_element], text=not binary).write(output_path)


num_faces = lambda mesh: len(mesh['face'].data)
num_verts = lambda mesh: len(mesh['vertex'].data)

def analyze_mesh_connectivity(original_mesh: PlyData, compressed_path: Path) -> None:
    """Analyze connectivity data of original and compressed mesh files"""
    
    if compressed_path.exists():
        compressed_size = compressed_path.stat().st_size
        print(f"Compressed connectivity size: {compressed_size} bytes")
        bpv, bpf = relative_conn_size(original_mesh, compressed_size)
        print(f"bpv | bpf: {round(bpv)} | {round(bpf)}")
    else:
        print(f"{compressed_path} not found")
    


def something():
    # bunny stats for nmax = 3, 4, 5
    stats = {
        "1k": {
            3: (.395, .11), # (efficiency, time)
            4: (.522, .5),
            5: (.576, 3)
        },
        "10k": {
            3: (.398, .62),
            4: (.531, 4),
            5: (.584, 39)
        },
        "70k": {
            3: (.417, 3.7),
            4: (.537, 31),
            5: (.580, 306)
        }
    }

    fig, ax = plt.subplots(nrows=1, ncols=len(stats))#, figsize=(15, 5))


    for i, (name, data) in enumerate(stats.items()):
        ax[i].set_title(f"{name} triangles")
        ax[i].set_xlabel(r"$n_{max}$", fontsize=15)
        # ax[i].set_ylabel(r"$\delta^r(W)$", rotation=0)
        # ax[i].set_ylabel(r"%", rotation=0)
        
        # eff = [v[0] for v in data.values()]
        eff = [100 * (1 - v[0]) for v in data.values()] # compression ratio
        ax[i].bar(data.keys(), eff, color='gray')
        # ax[i].set_ylim(0, 1)
        ax[i].set_ylim(0, 100)
        ax[i].set_xticks([3, 4, 5])
        ax[i].set_xticklabels([3, 4, 5])

        for j, v in enumerate(eff):
            # ax[i].text(list(data.keys())[j], v + 0.02, f"{v:.3f}", ha='center')
            ax[i].text(list(data.keys())[j], v + 0.02 * 100, f"{v:.1f}%", ha='center', fontsize=13)

        time = [v[1] for v in data.values()]

        for j, v in enumerate(time):
            t = round(v, 1) if v < 5 else round(v)
            ax[i].text(list(data.keys())[j], 0.05 * 100, f"{t} s", ha='center', color="white", fontsize=13)
 
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # print_plyfiles('data/MeshLab_sample_meshes')

    test_meshes = "bunny2 bunny10k bunny70k screwdriver".split()

    for mesh_name in test_meshes:
        mesh_path = Path(f"data/MeshLab_sample_meshes/{mesh_name}.ply")
        plydata = PlyData.read(mesh_path)
        n_faces = num_faces(plydata)
        n_verts = num_verts(plydata)

        print(f"Mesh: {mesh_name} | Faces: {n_faces} | Vertices: {n_verts}")
        continue
        compressed_path = Path(mesh_path).parent / f"{mesh_name}_AtlasCompressed" / f"{mesh_name}_3-gons.conn.zst"

        print(f"Mesh: {mesh_name}")
        analyze_mesh_connectivity(plydata, compressed_path)
        print("-" * 50)
        # conn_path = Path(f"data/MeshLab_sample_meshes/wireframes/{mesh_name}_conn_only.ply.zst")
        # conn_size = conn_path.stat().st_size
        # print(mesh_name, *(round(sz) for sz in relative_conn_size(plydata, conn_size)))