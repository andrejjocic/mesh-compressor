import networkx as nx
from typing import List, Tuple, TypeAlias, Set, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
import functools
import network_utils as net
import itertools

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
        # (simple vertex-to-vertex mapping; make sure you update it if cycles are added!)
        for cycle in self.cycles:
            if v in cycle:
                i = cycle.index(v)
                return cycle[(i + 1) % len(cycle)]
        
        return v # fixed points are not stored
    
    def apply_edge(self, e: Edge) -> Edge:
        u, v = e
        return (self.apply(u), self.apply(v))
    
    @functools.cached_property
    def size(self) -> int:
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
    def size(self) -> int:
        """compressed graph size, as number of edges"""
        return len(self.G_diff_H) + self.H_resid.number_of_edges() + self.perm.size
    
    def relative_efficiency(self, full_size: int) -> float:
        """full graph size given as number of edges"""
        return (full_size - self.size) / full_size


def compress_graphlets(G):
    # [ČM21] algorithm 1
    raise NotImplementedError()

def compress_bipartite(G: nx.Graph) -> NSCompressedGraph:
    # [ČM21] algorithm 2
    G = G.copy() # OPT: modify in place for benchmarking

    def rel_efficiency(g: int, u: int, v: int) -> float:
        e = (2*g - v - u + 1 - u*v) / g
        assert e <= 1
        return e
    
    def extract_bipart(v0: int) -> Tuple[List[Edge], List[Edge], float]:
        """returns (U, V, efficiency)"""
        U = set(G.neighbors(v0))
        if len(U) < 2: return [], [], 0.0

        V = set()
        for u in U:
            V.update(G.neighbors(u))
        V -= U
        if len(V) < 2: return [], [], 0.0
        
        assert v0 in V
        assert len(U) > 0 and len(V) > 0
        assert not U & V
        # G(U,V) is the bipartite subgraph of G in vertex sets U and V
        Guv = nx.Graph(G.subgraph(U | V)) # OPT: try to avoid copying
        Guv.remove_edges_from((u1, u2) for u1, u2 in itertools.combinations(U, 2) if G.has_edge(u1, u2))
        Guv.remove_edges_from((v1, v2) for v1, v2 in itertools.combinations(V, 2) if G.has_edge(v1, v2))
        assert nx.is_bipartite(Guv)

        assert Guv.number_of_edges() <= len(U) * len(V)
        
        max_eff = rel_efficiency(Guv.number_of_edges(), len(U), len(V))
        improving = True

        def greedy_optimize(A: Set[int], B: Set[int]) -> None:
            """try to remove a vertex from A to improve efficiency of G(A,B)"""
            nonlocal max_eff, improving
            for a in A:
                Na = list(Guv.neighbors(a)) ;assert all(b in B for b in Na)
                edges_left = Guv.number_of_edges() - len(Na)
                now_isolated = [b for b in Na if Guv.degree[b] == 1] ;assert len(now_isolated) < len(B)

                if (eff := rel_efficiency(edges_left, len(A) - 1, len(B) - len(now_isolated))) > max_eff:
                    Guv.remove_node(a)
                    Guv.remove_edges_from((a, b) for b in now_isolated);assert Guv.number_of_edges() == edges_left
                    # should we also remove the isolated vertices from Guv?
                    A.remove(a)
                    B.difference_update(now_isolated)
                    max_eff = eff
                    improving = True
                    return

        while improving and len(U) + len(V) > 2 + 2: # u=v=2 is the minimum NSC graph (is this a redundant check?)
            improving = False
            if len(U) > 2:
                greedy_optimize(U, V)
            if len(V) > 2:
                greedy_optimize(V, U)
                    
        # print(f"extracted bipartite subgraph with efficiency {max_eff:.2f}")
        return U, V, max_eff # OPT: also return the removed edges (so we don't have to loop K(U,V) in below While body)


    edge_diff: List[Edge] = []
    vertex_perm = VertexPermutation([])

    while True:
        U, V, effic = max((extract_bipart(v) for v in G.nodes), key=lambda UVe: UVe[2])
        if effic <= 0: break

        for u, v in itertools.product(U, V):
            if G.has_edge(u, v): # FIXME: there was a case where u,v in G but not in residual
                G.remove_edge(u, v)
        # NOTE: do we need to worry about disconnecting the graph?
        #       (may at least save time by removing isolated vertices from further consideration)
            
        # G(U,V) edges removed from G, remaining K(U,V) edges added to G (to validate permutation)
        edge_diff.extend(itertools.product(U, V)) 
        vertex_perm.cycles.append(U) # explicitly encode only edges (U[0], V[i]) for i > 0
        # NOTE: can we improve efficiency by simply swapping U and V? (appending minimal cycle)
        # (probably balances out with the edges we remove from the residual graph)
        print(f"extracted bipartite subgraph with efficiency {effic:.2f}")

    return NSCompressedGraph(edge_diff, H_resid=G, perm=vertex_perm)


def decomp_square():
    # [ČM21] table 1
    G_resid = nx.cycle_graph(4)
    G_resid = nx.convert_node_labels_to_integers(G_resid, first_label=1)
    G_resid.remove_edge(2, 3)
    net.draw_graph(G_resid, "residual with 3 edges")
    G_decomp = decompress_SC(G_resid, VertexPermutation([[1,2], [4,3]]))
    net.draw_graph(G_decomp, "decomp from 3 edges")
    G_resid.remove_edge(4, 3)
    net.draw_graph(G_resid, "residual with 2 edges")
    G_decomp = decompress_SC(G_resid, VertexPermutation([[1,3]]))
    net.draw_graph(G_decomp, "decomp from 2 edges")
    G_resid.remove_edge(1, 4)
    net.draw_graph(G_resid, "residual with 1 edge")
    G_decomp = decompress_SC(G_resid, VertexPermutation([[1,2,3,4]]))
    net.draw_graph(G_decomp, "decomp from 1 edge")


def decomp_NSC_example():
    # [ČM21] fig. 2
    Hpi = nx.Graph()
    Hpi.add_edges_from((int(e[0]), int(e[1])) for e in "12 14 15 16 17 23 26 27 67".split())
    assert Hpi.number_of_edges() == 9
    net.draw_graph(Hpi, "H_pi (residual)")

    G = NSCompressedGraph(
        G_diff_H=[(6,3)], # will be removed from H to make G
        H_resid=Hpi,
        perm=VertexPermutation([[1,4], [2,3]])
    ).decompress()

    net.draw_graph(G, "decompressed NSC")
    assert G.number_of_edges() == 14


if __name__ == "__main__":
    G = net.read_pajek("karate_club", data_folder="data\\networks")
    print(G)
    C = compress_bipartite(G)
    print(f"compressed to size {C.size}, rel.effic.={C.relative_efficiency(G.number_of_edges()):.4f}")
    # TODO: ensure properly decompressed