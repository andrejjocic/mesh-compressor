import networkx as nx
from typing import List, Tuple, TypeAlias, Set, Dict, Optional, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import functools
import network_utils as net
import itertools
import heapq
from enum import Enum, auto
from collections import defaultdict
from tqdm import tqdm
import struct

Edge: TypeAlias = Tuple[int, int]
Cycle: TypeAlias = List[int]


def eq_edges(e1: Edge, e2: Edge) -> bool:
    u, v = e1
    return e2 == (u, v) or e2 == (v, u)


@dataclass
class VertexPermutation:
    """graph vertex permutation, in cycle notation"""

    cycles: List[Cycle] # TODO: refactor to use sympy.combinatorics.Permutation

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
        # The permutation can be ‘applied’ to any list-like object, not only Permutations:
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
            if i > 1000: raise Exception("infinite loop in decompression?") # TODO: delete this
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

        # print(f"unzipped H, then removed {removed}, added {added}")
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
    
    def append_to_file(self, filename: str) -> None:
        with open(filename, "a") as file:
            file.write(f"uncompressed {self.uncompressed.number_of_edges()}\n")
            for u, v in self.uncompressed.edges:
                file.write(f"{u} {v}\n")

            file.write(f"compressed {len(self.compressed)}\n")
            for C in self.compressed:
                file.write(f"residual {C.H_resid.number_of_edges()}\n")
                for u, v in C.H_resid.edges:
                    file.write(f"{u} {v}\n")
                file.write(f"perm {len(C.perm.cycles)}\n")
                for cycle in C.perm.cycles:
                    file.write(" ".join(map(str, cycle)) + "\n")
                file.write(f"diff {len(C.G_diff_H)}\n")
                for u, v in C.G_diff_H:
                    file.write(f"{u} {v}\n")


    def serialize(self, filename: str) -> None:
        # OPT: some sort of buffering
        with open(filename, "wb") as file:
            file.write(struct.pack("I", self.uncompressed.number_of_edges()))
            for u, v in self.uncompressed.edges:
                file.write(struct.pack("II", u, v))

            file.write(struct.pack("I", len(self.compressed)))
            for C in self.compressed:
                file.write(struct.pack("I", C.H_resid.number_of_edges()))
                for u, v in C.H_resid.edges:
                    file.write(struct.pack("II", u, v))
                file.write(struct.pack("I", len(C.perm.cycles)))
                for cycle in C.perm.cycles:
                    file.write(struct.pack("I", len(cycle)))
                    for v in cycle:
                        file.write(struct.pack("I", v))
                file.write(struct.pack("I", len(C.G_diff_H)))
                for u, v in C.G_diff_H:
                    file.write(struct.pack("II", u, v))



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


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    vaild: bool
    item: Any=field(compare=False)

    def __init__(self, priority: float, item: Any):
        self.priority = priority
        self.item = item
        self.vaild = True


class CachingMode(Enum):
    NONE = auto()
    """don't cache subgraphs, recompute them every time"""
    STATIC = auto()
    """cache subgraphs at the start, but don't update the cache (only invalidate entries)"""
    DYNAMIC = auto()
    """cache subgraphs at the start, then recompute optimal subgraphs from affected vertices after each compression"""

    @property
    def enabled(self) -> bool:
        return self != CachingMode.NONE


def compress_bipartite(G: nx.Graph, caching_mode=CachingMode.DYNAMIC) -> NSCompressedPartition:
    # [ČM21] algorithm 2
    Gcomp = NSCompressedPartition(G) # OPT: modify G in place (for proper benchmarking)
    subgraph_stats = defaultdict(list)

    if caching_mode.enabled:
        subgraph_cache: List[PrioritizedItem] = []
        for effic, Guv, U, V in (extract_bipart(Gcomp.uncompressed, v) for v in G.nodes):
            if effic > 0: # don't bother caching subgraphs that won't be selected
                subgraph_cache.append(PrioritizedItem(priority=-effic, item=(Guv, U, V))) # sort by efficiency (descending)

        heapq.heapify(subgraph_cache)
        # NOTE: with static caching we only need a pre-sorted list (not heap), but asymptotic complexity is nlgn in any case
        pbar = tqdm(total=len(subgraph_cache), desc="processing subgraphs")

    while True:
        if caching_mode == CachingMode.NONE:
            effic, Guv, U, V = max((extract_bipart(Gcomp.uncompressed, v) for v in G.nodes), key=lambda res: res[0])
            if effic <= 0: break
        else:
            try:
                best = heapq.heappop(subgraph_cache)
                pbar.update(1)
                if not best.vaild: continue

                Guv, U, V = best.item
                effic = -best.priority
                assert effic > 0
            except IndexError:
                break # no more subgraphs to compress
       
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
        if len(Guv_comp.G_diff_H) > 0: print(f"completing to K(U,V) with {Guv_comp.G_diff_H}")
        # assert Guv_comp.relative_efficiency(Guv.number_of_edges()) == effic
        Gcomp.compressed.append(Guv_comp)
        subgraph_stats[tuple(sorted([len(U), len(V)]))].append(effic)

        # invalidate cache entries intersecting the removed subgraph
        if caching_mode.enabled and len(subgraph_cache) > 0:
            invalidated = 0
            for entry in subgraph_cache: # OPT: maintin map from edges to cache entries, loop only G(U,V) edges here
                if edge_intersect(Guv, entry.item[0]):
                    entry.vaild = False
                    invalidated += 1

            # if invalidated > 0:
            #     print(f"invalidated {invalidated} cache entries ({100 * invalidated / len(subgraph_cache):.2f}%)")

            if caching_mode == CachingMode.DYNAMIC:
                for x in Guv.nodes:
                    effic, Guv, U, V = extract_bipart(Gcomp.uncompressed, x)
                    if Gcomp.uncompressed.degree[x] == 0: assert effic == 0
                    if effic > 0:
                        heapq.heappush(subgraph_cache, PrioritizedItem(priority=-effic, item=(Guv, U, V)))
                        pbar.total += 1

    if caching_mode.enabled: pbar.close()
    for uv, effs in subgraph_stats.items():
        print(f"extracted {len(effs)}x G{uv}, avg eff = {sum(effs) / len(effs):.3f}, max eff = {max(effs):.3f}")
    return Gcomp


def edge_intersect(G1: nx.Graph, G2: nx.Graph) -> bool:
    """check if two graphs have a non-empty edge intersection"""
    return any(e in G2.edges for e in G1.edges)




if __name__ == "__main__":
    # graph = nx.gnp_random_graph(n=30, p=0.5)
    # graph = net.read_pajek("karate_club", data_folder="data\\networks")
    graph = nx.erdos_renyi_graph(n=50, p=0.9)
    # print(graph)
    for caching_mode in [CachingMode.STATIC, CachingMode.DYNAMIC]:
        # print(f"compressing with caching mode {caching_mode}")
        C = compress_bipartite(graph, caching_mode)
        # print(C)
        # print("decompressing...")
        # print(C.decompress())
        # print()