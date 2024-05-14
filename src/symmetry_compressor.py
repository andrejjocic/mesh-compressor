import networkx as nx
from typing import List, Tuple, TypeAlias
from dataclasses import dataclass
import matplotlib.pyplot as plt
import functools

Edge: TypeAlias = Tuple[int, int]
Cycle: TypeAlias = List[int]


def eq_edges(e1: Edge, e2: Edge) -> bool:
    u, v = e1
    return e2 == (u, v) or e2 == (v, u)


@dataclass
class VertexPermutation:
    """graph vertex permutation, in cycle notation"""

    cycles: List[Cycle]

    def apply(self, v: int) -> int:
         # OPT: optimize this if needed for compression (likely won't bother benchmarking decompression)
        for cycle in self.cycles:
            if v in cycle:
                i = cycle.index(v)
                return cycle[(i + 1) % len(cycle)]
        
        return v # fixed points are not stored
    
    def apply_edge(self, e: Edge) -> Edge:
        u, v = e
        return (self.apply(u), self.apply(v))
    
    @functools.cached_property
    def __len__(self) -> int:
        """length of the permutation, as number of edges"""
        return sum(len(c) - 1 for c in self.cycles)
    

def decompress_SC(G_residual: nx.Graph, perm: VertexPermutation) -> nx.Graph:
    """decompress a symmetry-compressed graph"""
    G = G_residual.copy() # OPT: let this be modified in place (for proper benchmarking)
    # take every edge in the residual graph and apply permutation
    # to both ends of it, until one cycle of the *edge* permutation is obtained
    for e in G_residual.edges:
        e_perm = e
        while True:
            e_perm = perm.apply_edge(e_perm)
            if eq_edges(e_perm, e):
                break
            G.add_edge(*e_perm)

    return G


@dataclass
class NSCompressedGraph:
    """near symmetry-compressed graph G"""

    G_diff_H: List[Edge]
    """symmetric difference of edges between G and H"""
    H_resid: nx.Graph
    """residual graph of H"""
    perm: VertexPermutation
    """permutation of the vertices of H (and therefore of G)"""

    def decompress(self) -> nx.Graph:
        H = decompress_SC(self.H_resid, self.perm)
        for e in self.G_diff_H:
            if e in H.edges:
                # print(f"removing edge {e}")
                H.remove_edge(*e)
            else:
                # print(f"adding edge {e}")
                H.add_edge(*e)

        return H
    
    @functools.cached_property
    def __len__(self) -> int:
        return len(self.G_diff_H) + self.H_resid.number_of_edges() + len(self.perm)
    
    def relative_efficiency(self, full_size: int) -> float:
        """full graph size given as number of edges"""
        return (full_size - len(self)) / full_size


def compress_graphlets(G):
    # [ČM21] algorithm 1
    raise NotImplementedError()

def compress_bipartite(G):
    # [ČM21] algorithm 2
    raise NotImplementedError()


def draw_graph(G: nx.Graph, title="", **kwargs):
    plt.title(f"{G.name} {title}")
    nx.draw(G, with_labels=True, **kwargs)
    plt.show()

def decomp_square():
    # [ČM21] table 1
    G_resid = nx.cycle_graph(4)
    G_resid = nx.convert_node_labels_to_integers(G_resid, first_label=1)
    G_resid.remove_edge(2, 3)
    draw_graph(G_resid, "residual with 3 edges")
    G_decomp = decompress_SC(G_resid, VertexPermutation([[1,2], [4,3]]))
    draw_graph(G_decomp, "decomp from 3 edges")
    G_resid.remove_edge(4, 3)
    draw_graph(G_resid, "residual with 2 edges")
    G_decomp = decompress_SC(G_resid, VertexPermutation([[1,3]]))
    draw_graph(G_decomp, "decomp from 2 edges")
    G_resid.remove_edge(1, 4)
    draw_graph(G_resid, "residual with 1 edge")
    G_decomp = decompress_SC(G_resid, VertexPermutation([[1,2,3,4]]))
    draw_graph(G_decomp, "decomp from 1 edge")


def decomp_NSC_example():
    # [ČM21] fig. 2
    Hpi = nx.Graph()
    Hpi.add_edges_from((int(e[0]), int(e[1])) for e in "12 14 15 16 17 23 26 27 67".split())
    assert Hpi.number_of_edges() == 9
    draw_graph(Hpi, "H_pi (residual)")

    G = NSCompressedGraph(
        G_diff_H=[(6,3)], # will be removed from H to make G
        H_resid=Hpi,
        perm=VertexPermutation([[1,4], [2,3]])
    ).decompress()

    draw_graph(G, "decompressed NSC")
    assert G.number_of_edges() == 14


if __name__ == "__main__":
    decomp_NSC_example()