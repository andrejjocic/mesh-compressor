import networkx as nx
from sympy.combinatorics import Permutation, generators
from dataclasses import dataclass
from pathlib import Path
from symmetry_compressor import eq_edges
from tqdm import tqdm
from typing import *
from collections import defaultdict
import networkx.algorithms.isomorphism as iso
import json

@dataclass
class SCGraphlet:
    """Symmetry-compressed graphlet"""
    atlas_index: int
    """index in the graph atlas"""
    compressive_auto_rank: int
    """rank of the most compressive automorphism"""

    def expand(self, index_map: ..., atlas: List[nx.Graph]) -> nx.Graph:
        pass


def repr_size(perm: Permutation) -> int:
    """size of permutation, as number of edges (ČM21 encoding)"""
    s = 0
    for cycle_len, mult in perm.cycle_structure.items():
        if cycle_len > 1:
            s += (cycle_len - 1) * mult
    return s


def cache_SC_graphlets(min_size=4, max_size=7, cache_dir="cache"):
    """cache symmetry-compressed representation of graphlets (one file for each size)"""
    assert max_size <= 7

    atlas = nx.graph_atlas_g() # all graphs up to 7 nodes, sorted by number of nodes
    i = 0
    while atlas[i].number_of_nodes() < min_size:
        i += 1 # OPT: figure out how to skip to min_size directly

    while atlas[i].number_of_nodes() <= max_size:
        G = atlas[i]

    raise NotImplementedError("TODO copy code from ipynb")


@dataclass
class AtlasGraphlet:
    """Graphlet from the graph atlas"""
    index: int
    """index in the graph atlas"""
    efficiency: float
    """compression efficiency of the graphlet"""
    # TODO: just add the atlas as an attribute?

    def expand(self, node_mapping: Dict[int,int], atlas: List[nx.Graph]) -> nx.Graph:
        """expand the graphlet to a full graph
        ### Arguments
        - node_mapping: mapping from output nodes to 0,1,2,..,n-1
        - atlas: list of graphs in the atlas
        """
        G = atlas[self.index]
        assert set(node_mapping.values()) == set(range(len(G.number_of_nodes())))
        return nx.relabel_nodes(G, {old: new for new, old in node_mapping.items()}, copy=True) # don't modify the atlas!
    


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

def compress_subgraphlets(G: nx.Graph, max_graphlet_sz=7, sort_by_efficiency=True) -> ...:
    # adapted version of [ČM21] algorithm 1
    if max_graphlet_sz > 7:
        raise NotImplementedError("only graphlets up to size 7 are supported")

    graphlet_atlas = nx.graph_atlas_g()
    G = G.copy() # OPT: modify G in place (for proper benchmarking)
    graphlets = load_atlas_efficiency(n_max=max_graphlet_sz)
    if sort_by_efficiency:
        graphlets.sort(key=lambda g: g.efficiency, reverse=True)

    compressed_subgraphs = []

    for graphlet_id in tqdm(graphlets, desc="looping over graphlets"):
        graphlet = graphlet_atlas[graphlet_id.index]
        matcher = iso.GraphMatcher(G, G2=graphlet, node_match=None, edge_match=None) # node and edge attributes are ignored
        # TODO: find find a *maximal* set of edge-disjoint isomorphic subgraphs
        for isomorphism in matcher.subgraph_isomorphisms_iter():
            # for subgraph on {0,1,2,3}, an isomorphism looks something like {4: 0, 13: 1, 2: 2, 44: 3}
            inv_iso = {v: k for k, v in isomorphism.items()}
            subgraph = nx.relabel_nodes(graphlet, mapping=inv_iso, copy=True)
            if any(not G.has_edge(*edge) for edge in subgraph.edges):
                continue # a part of this subgraph has already been compressed (with the same graphlet)
            
            # compress the subgraph
            mapped_nodes = [inv_iso[i] for i in range(subgraph.number_of_nodes())]
            compressed_subgraphs.append((graphlet_id, mapped_nodes))
            G.remove_edges_from(subgraph.edges)

    return G, compressed_subgraphs



if __name__ == "__main__":
    # cache_atlas_efficiency()
    G = nx.erdos_renyi_graph(n=20, p=0.75)
    G_resid, compressed_subgraphs = compress_subgraphlets(G, max_graphlet_sz=4)

    orig_size = 2 * G.number_of_edges() # measured as number of vertex indices
    atlas = nx.graph_atlas_g()
    compressed_size = 2 * G_resid.number_of_edges() + sum(1 + atlas[graphlet.index].number_of_nodes() for graphlet, _ in compressed_subgraphs)

    print(f"original size: {orig_size}, compressed size: {compressed_size}, relative efficiency: {1 - compressed_size / orig_size:.3f}")