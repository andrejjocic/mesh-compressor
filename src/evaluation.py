from typing import *
from pathlib import Path
from mesh_utils import skeleton
from plyfile import PlyData
from symmetry_compressor import compress_bipartite, CachingMode

def compression_ratio(original_size, compressed_size):
    return original_size / compressed_size

# def weissman_score()



if __name__ == '__main__':
    # folder = 'data/MeshLab_sample_meshes'
    # plyfiles = Path(folder).rglob('*.ply')
    # sorted_plyfiles = sorted(plyfiles, key=lambda x: x.stat().st_size)
    # for plyfile in sorted_plyfiles:
    #     print(plyfile, plyfile.stat().st_size)
    graph, _ = skeleton(PlyData.read(r"data\MeshLab_sample_meshes\T.ply"))
    print(graph)
    for caching_mode in [CachingMode.STATIC, CachingMode.DYNAMIC]:
        print(f"compressing with caching mode {caching_mode}")
        C = compress_bipartite(graph, caching_mode)
        print(C)