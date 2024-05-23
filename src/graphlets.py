import networkx as nx
from sympy.combinatorics import Permutation, generators
from dataclasses import dataclass
from pathlib import Path
from symmetry_compressor import eq_edges
from tqdm import tqdm
from typing import *
from collections import defaultdict, Counter
import networkx.algorithms.isomorphism as iso
import json
import network_utils as net
import struct


def repr_size(perm: Permutation) -> int:
    """size of permutation, as number of edges (ČM21 encoding)"""
    s = 0
    for cycle_len, mult in perm.cycle_structure.items():
        if cycle_len > 1:
            s += (cycle_len - 1) * mult
    return s


@dataclass
class AtlasGraphlet:
    """Identifier of a graphlet from the graph atlas"""
    index: int
    """index in the graph atlas"""
    efficiency: float
    """compression efficiency of the graphlet"""
    # just add the atlas as an attribute?

    def expand(self, node_mapping: List[int], atlas: List[nx.Graph]) -> nx.Graph:
        """expand the graphlet to a full graph
        ### Arguments
        - node_mapping: the i-th node in the graphlet is mapped to node_mapping[i]-th node in the full graph
        - atlas: list of graphs in the atlas
        """
        G = atlas[self.index]
        assert len(node_mapping) == G.number_of_nodes()
        return nx.relabel_nodes(G, dict(enumerate(node_mapping)), copy=True) # don't modify the atlas!
    


def cache_atlas_efficiency(cache_dir="cache"):
    n2graphlets = defaultdict(list)
    atlas = nx.graph_atlas_g()

    for atlas_idx, G in enumerate(tqdm(atlas)):
        if atlas_idx == 0 or not nx.is_connected(G):
            continue # NOTE: approach may be extensible to disconnected graphs

        n = G.number_of_nodes()
        m = G.number_of_edges()

        if 1 + n < 2 * m:
            rel_eff = 1 - (1 + n) / (2 * m)
            assert 0 < rel_eff <= 1
            n2graphlets[n].append(AtlasGraphlet(atlas_idx, rel_eff))
        elif m > 0:
            assert 1 - (1 + n) / (2 * m) <= 0
     # OPT: since altlas is (secondarily) sorted by m, we can break early (skip to next n)

    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir()

    for n, graphlets in n2graphlets.items():
        path = cache_dir / f"atlas_efficiency_{n}.json"
        with path.open("w") as f:
            json.dump([graphlet.__dict__ for graphlet in graphlets], f)
            # TODO: sort by efficiency right here? (don't even need to store efficiencies then..)

        # print(f"n={n}: {len(effs)} compressive, mean eff = {sum(effs) / len(effs):.3f}, max eff = {max(effs):.3f}")


def load_atlas_efficiency(n_min=3, n_max=7, cache_dir="cache") -> List[AtlasGraphlet]:
    cache_dir = Path(cache_dir)
    graphlets = []
    for n in range(n_min, n_max+1):
        path = cache_dir / f"atlas_efficiency_{n}.json"
        if path.exists():
            with path.open("r") as f:
                graphlets.extend([AtlasGraphlet(**d) for d in json.load(f)])
        else:
            print(f"Warning: file {path} not found!")

    return graphlets


class AtlasCompressedGraph:
    """Graph compressed using the graph atlas"""

    residual: nx.Graph
    """the uncompressed edges of the graph"""
    full_size: int
    """size of the full graph, as number of vertex indices"""
    compressed_graphlets: List[Tuple[AtlasGraphlet, List[int]]]
    """compressed subgraphs, as a list of graphlet IDs and node mappings"""
    atlas: Optional[List[nx.Graph]]
    """optional reference to the graph atlas"""

    def __init__(self, G: nx.Graph, take_ownership=False, atlas: Optional[List[nx.Graph]] = None):
        self.residual = G if take_ownership else G.copy()
        self.full_size = 2 * G.number_of_edges()
        self.compressed_graphlets = []
        self.atlas = atlas

    def __repr__(self) -> str:
        if self.full_size is None: # deserialized
            return f"AtlasCompressed({self.residual.name})"
        else:
            return f"AtlasCompressed({self.residual.name}, eff={self.relative_efficiency:.3f})"

    def compress(self, subgraph: nx.Graph, graphlet: AtlasGraphlet, mapped_nodes: List[int]):
        """replace the subgraph with a compact representation"""
        assert self.atlas[graphlet.index].number_of_nodes() == len(mapped_nodes)

        self.residual.remove_edges_from(subgraph.edges)
        self.compressed_graphlets.append((graphlet, mapped_nodes))
        # TODO: try the other encoding if graphlets often reoccur
        # (we can easily afford extra byte for a switch flag, so decide based on efficiency)

    def decompress(self, in_place=True) -> nx.Graph:
        graphlet_atlas = nx.graph_atlas_g() if self.atlas is None else self.atlas
        full = self.residual if in_place else self.residual.copy()

        for graphlet_id, mapped_nodes in self.compressed_graphlets:
            full.add_edges_from(graphlet_id.expand(mapped_nodes, graphlet_atlas).edges)

        return full
    
    @property
    def relative_efficiency(self) -> float:
        return (self.full_size - self.size) / self.full_size
    
    @property
    def size(self) -> int:
        """size of the compressed graph, as number of vertex indices"""
        return 2 * self.residual.number_of_edges() + sum(1 + len(nodemap) for _, nodemap in self.compressed_graphlets)


    def serialize(self, filename: str) -> None:
        # OPT: some sort of buffering
        with open(filename, "wb") as f:
            # encode the residual graph
            f.write(struct.pack("<I", self.residual.number_of_edges()))
            for edge in self.residual.edges:
                f.write(struct.pack("<II", *edge))

            # encode the compressed graphlets
            f.write(struct.pack("<I", len(self.compressed_graphlets)))
            for graphlet, nodemap in self.compressed_graphlets:
                f.write(struct.pack("<I", graphlet.index))
                # we can infer the length of the nodemap from the graphlet
                for node in nodemap:
                    f.write(struct.pack("<I", node))



    @staticmethod
    def deserialize(filename: str) -> "AtlasCompressedGraph":
        filename = Path(filename)
        SIZEOF_INT = 4 # bytes

        with open(filename, "rb") as f:
            # unpack the residual graph
            G = nx.Graph(name=f"deserialized_{filename.stem}")
            num_edges, = struct.unpack("<I", f.read(SIZEOF_INT))
            for _ in range(num_edges):
                u, v = struct.unpack("<II", f.read(2 * SIZEOF_INT))
                G.add_edge(u, v)

            acg = AtlasCompressedGraph(G, take_ownership=True, atlas=nx.atlas.graph_atlas_g())
            acg.full_size = None
            
            # unpack the compressed graphlets
            num_graphlets, = struct.unpack("<I", f.read(SIZEOF_INT))
            for _ in range(num_graphlets): # pbar?
                graphlet_idx, = struct.unpack("<I", f.read(SIZEOF_INT))
                n = acg.atlas[graphlet_idx].number_of_nodes()
                nodemap = struct.unpack("<" + "I"*n, f.read(n * SIZEOF_INT))
                acg.compressed_graphlets.append((
                    AtlasGraphlet(graphlet_idx, efficiency=None),
                    list(nodemap)
                ))

        return acg
        


def compress_subgraphlets(G: nx.Graph, max_graphlet_sz=7, sort_by_efficiency=True, print_stats=False) -> AtlasCompressedGraph:
    # adapted version of [ČM21] algorithm 1
    if max_graphlet_sz > 7:
        raise NotImplementedError("only graphlets up to size 7 are supported")

    graphlet_atlas = nx.graph_atlas_g()
    Gcomp = AtlasCompressedGraph(G, take_ownership=False, atlas=graphlet_atlas)
    graphlets = load_atlas_efficiency(n_max=max_graphlet_sz) # TODO: option to halt at certain efficiency threshold
    if sort_by_efficiency: 
        graphlets.sort(key=lambda g: g.efficiency, reverse=True) # OPT: just have the cache sorted by efficiency

    graphlet_stats = Counter()

    for graphlet_id in tqdm(graphlets, desc="looping over graphlets", ncols=100):
        graphlet = graphlet_atlas[graphlet_id.index]
        matcher = iso.GraphMatcher(Gcomp.residual, G2=graphlet, node_match=None, edge_match=None) # node and edge attributes are ignored

        for isomorphism in matcher.subgraph_isomorphisms_iter():
            # for subgraph on {0,1,2,3}, an isomorphism looks something like {4: 0, 13: 1, 2: 2, 44: 3}
            inv_iso = {v: k for k, v in isomorphism.items()}
            subgraph = nx.relabel_nodes(graphlet, mapping=inv_iso, copy=True)
            if any(not Gcomp.residual.has_edge(*edge) for edge in subgraph.edges):
                # print(f"skipping instance of graphlet {graphlet_id.index} (not edge-disjoint)")
                continue # a part of this subgraph has already been compressed (with the same graphlet)
            
            mapped_nodes = [inv_iso[i] for i in range(subgraph.number_of_nodes())]
            Gcomp.compress(subgraph, graphlet_id, mapped_nodes)
            graphlet_stats[graphlet_id.index] += 1

    if print_stats:
        print("Graphlet stats (#occurences of <index>):")
        for idx, count in sorted(graphlet_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"{count}x {idx}")

    return Gcomp



if __name__ == "__main__":
    # cache_atlas_efficiency()
    # G = nx.erdos_renyi_graph(n=20, p=0.75)
    G = nx.Graph(net.read_pajek("karate_club", data_folder="data\\networks"))
    Gcomp = compress_subgraphlets(G, max_graphlet_sz=6, sort_by_efficiency=True)
    print(Gcomp)

    # graphlet_idx2node_sets = defaultdict(list)
    # for graphlet, node_set in compressed_subgraphs:
    #     graphlet_idx2node_sets[graphlet.index].append(node_set)

    # for graphlet_idx, node_sets in graphlet_idx2node_sets.items():
    #     net.draw_graph(atlas[graphlet_idx], title=str(node_sets))

    bin_path = Path("karate_compressed.acgf")
    Gcomp.serialize(bin_path)

    Gcomp2 = AtlasCompressedGraph.deserialize(bin_path)
    print(Gcomp2)
    G2 = Gcomp2.decompress()
    print(G2)
    # print(set(G.edges) == set(G2.edges))
    # print(sorted(G.edges))
    # print(sorted(G2.edges))
    print(nx.is_isomorphic(G, G2))

    bin_path.unlink()