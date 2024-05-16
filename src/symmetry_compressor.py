import networkx as nx
from typing import List, Tuple, TypeAlias, Set, Dict, Optional
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
    
    def cycle_index(self, v: int) -> Optional[int]:
        for i, cycle in enumerate(self.cycles):
            if v in cycle:
                return i
        return None
    
    def apply_edge(self, e: Edge) -> Edge:
        u, v = e
        # if (ui := self.cycle_index(u)) is not None and ui == self.cycle_index(v):
        #     raise Exception(f"edge {e} vertices on the same cycle")
        
        return (self.apply(u), self.apply(v))
    
    @property # don't cache this property! (with current interface for adding cycles)
    def size(self) -> int:
        """length of the permutation, as number of edges"""
        return sum(len(c) - 1 for c in self.cycles)
    
    def assert_valid(self) -> None:
        """make sure the cycles are disjoint"""
        all_vertices = set(itertools.chain.from_iterable(self.cycles))
        assert len(all_vertices) == sum(map(len, self.cycles)), "cycles are not disjoint"
    

def decompress_SC(G_residual: nx.Graph, perm: VertexPermutation) -> nx.Graph:
    """decompress a symmetry-compressed graph"""
    G = G_residual.copy() # OPT: let this be modified in place (for proper benchmarking)
    # take every edge in the residual graph and apply permutation
    # to both ends of it, until one cycle of the *edge* permutation is obtained
    for e in G_residual.edges:
        e_perm = e
        i=0
        while True:
            i += 1
            if i > 100: raise Exception("infinite loop in decompression")
            e_perm = perm.apply_edge(e_perm)
            if eq_edges(e_perm, e):
                break
            G.add_edge(*e_perm)#; print(f"added edge {e_perm}")

    return G


@dataclass
class NSCompressedGraph:
    """near symmetry-compressed graph G"""

    G_diff_H: List[Edge]
    """symmetric difference of edges between G and H"""
    H_resid: nx.Graph
    """residual graph of H (the symmetry-compressible graph)"""
    perm: VertexPermutation
    """permutation of the vertices of H (and therefore of G)"""

    def __repr__(self) -> str:
        return f"NSCompressedGraph(resid={self.H_resid.number_of_edges()}, diff={len(self.G_diff_H)})"

    def decompress(self) -> nx.Graph:
        H = decompress_SC(self.H_resid, self.perm)
        removed, added = 0, 0
        for e in self.G_diff_H:
            if e in H.edges:
                H.remove_edge(*e); removed += 1
            else:
                H.add_edge(*e); added += 1

        print(f"unzipped H, then removed {removed}, added {added}")
        return H
    
    @property
    def size(self) -> int:
        """compressed graph size, as number of edges"""
        return len(self.G_diff_H) + self.H_resid.number_of_edges() + self.perm.size
    
    def relative_efficiency(self, full_size: int) -> float:
        """full graph size given as number of edges"""
        return (full_size - self.size) / full_size
    

class NSCompressedPartition:
    """partition of G into its NSCompressed subgraphs, + the rest of G"""
    
    uncompressed: nx.Graph
    """residual graph of G, after removing all NSCompressed subgraphs"""
    compressed: List[NSCompressedGraph]
    """list of NSCompressed subgraphs of G"""
    full_size: int
    """full size of G, as number of edges"""

    def __init__(self, G: nx.Graph):
        self.full_size = G.number_of_edges()
        self.compressed = []
        self.uncompressed = G.copy()


    def __repr__(self) -> str:
        return f"NSCompressed({self.uncompressed.name}, eff={self.relative_efficiency:.3f})"

    @property
    def size(self) -> int:
        """total size of the partition, as number of edges"""
        return self.uncompressed.number_of_edges() + sum(c.size for c in self.compressed)
    
    @property
    def relative_efficiency(self) -> float:
        return (self.full_size - self.size) / self.full_size
    
    def decompress(self) -> nx.Graph:
        G = self.uncompressed.copy()
        for C in self.compressed:
            G.add_edges_from(C.decompress().edges)

        return G


def compress_graphlets(G):
    # [ČM21] algorithm 1
    raise NotImplementedError()


def rel_efficiency(g: int, u: int, v: int) -> float:
    e = (2*g - v - u + 1 - u*v) / g # OPT: don't bohther with +1 if just optimizing this
    assert e <= 1
    return e

def extract_bipart(G: nx.Graph, v0: int) -> Tuple[float, nx.Graph, Set[Edge], Set[Edge]]:
    """returns efficiency, G(U,V), U, V"""
    U = set(G.neighbors(v0))

    V = set()
    for u in U:
        V.update(G.neighbors(u))
    V -= U
    if len(U) < 2 or len(V) < 2: return 0, None, None, None
    
    assert v0 in V; assert not U & V
    # G(U,V) is the bipartite subgraph of G in vertex sets U and V
    Guv = nx.Graph(G.subgraph(U | V)) # OPT: try to avoid copying
    Guv.remove_edges_from((u1, u2) for u1, u2 in itertools.combinations(U, 2) if G.has_edge(u1, u2))
    Guv.remove_edges_from((v1, v2) for v1, v2 in itertools.combinations(V, 2) if G.has_edge(v1, v2))
    assert nx.is_bipartite(Guv); assert Guv.number_of_edges() <= len(U) * len(V)
    
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
                Guv.remove_edges_from((a, b) for b in now_isolated) ;assert Guv.number_of_edges() == edges_left
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
                
    return max_eff, Guv, U, V


def compress_bipartite(G: nx.Graph) -> NSCompressedPartition:
    # [ČM21] algorithm 2
    Gcomp = NSCompressedPartition(G) # OPT: modify G in place (for proper benchmarking)

    while True: # TODO: progress bar showing proportion of compressed edges
        effic, Guv, U, V = max((extract_bipart(Gcomp.uncompressed, v) for v in G.nodes), key=lambda res: res[0])
        if effic <= 0: break
        assert Guv.number_of_edges() > (len(V) + len(U) - 1 + len(U) * len(V)) / 2 # [ČM21] Theorem 6

        # remove G(U,V) from G
        Gcomp.uncompressed.remove_edges_from(Guv.edges)

        # encode K(U,V)
        U = list(U); 
        Guv_comp = NSCompressedGraph(
            H_resid=nx.star_graph([U[0]] + list(V)), # the only fan of edges in K(U,V) that we explicitly encode
            perm=VertexPermutation([U]),
            # ensure K(U,V) (+) G(U,V) will be deleted at decompression
            G_diff_H=[(u, v) for u in U for v in V if not Guv.has_edge(u, v)] 
        )
        print(Guv_comp)
        assert Guv_comp.relative_efficiency(Guv.number_of_edges()) == effic
        Gcomp.compressed.append(Guv_comp)
        print(f"removed K({len(U)},{len(V)}) eff = {effic:.3f}")

    return Gcomp


if __name__ == "__main__":
    pass