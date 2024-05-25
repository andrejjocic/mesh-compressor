from typing import *
from pathlib import Path
from mesh_utils import skeleton
from plyfile import PlyData
from symmetry_compressor import compress_bipartite, CachingMode
from collections import defaultdict

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


if __name__ == '__main__':
    print_plyfiles('data/MeshLab_sample_meshes')

    # graph, _ = skeleton(PlyData.read(r"data\MeshLab_sample_meshes\T.ply"))
    # print(graph)
    # for caching_mode in [CachingMode.STATIC, CachingMode.DYNAMIC]:
    #     print(f"compressing with caching mode {caching_mode}")
    #     C = compress_bipartite(graph, caching_mode)
    #     print(C)