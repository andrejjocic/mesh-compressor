import networkx as nx
from typing import List, Tuple, TypeAlias, Set, Dict, Optional, Any, Iterator
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
from sympy.combinatorics import Permutation
from sympy.combinatorics.generators import symmetric
import networkx.algorithms.isomorphism as iso
import json
import os
from pathlib import Path


Edge: TypeAlias = Tuple[int, int]
Cycle: TypeAlias = List[int]


def eq_edges(e1: Edge, e2: Edge) -> bool:
    u, v = e1
    return e2 == (u, v) or e2 == (v, u)


@dataclass
class VertexPermutation:
    """graph vertex permutation, in cycle notation (fixed points are not stored)"""
    # FIXME: how do you infer range if fixed point not stored? (we can live without range error checking)

    cycles: List[Cycle]

    def __post_init__(self):
        self.assert_valid()

    @property
    def nontrivial_cycles(self) -> int:
        """number of non-singleton cycles"""    
        return len(self.cycles)


    def __repr__(self) -> str:
        return f"VxPerm({self.cycles})"

    def __call__(self, x) -> int:
        if isinstance(x, tuple) and len(x) == 2:  # Check if x is an edge (2-tuple)
            return self.apply_edge(x)
        elif isinstance(x, int):
            return self.apply(x)
        else:
            raise TypeError(f"can't apply VertexPermutation to type {type(x)}")

    def apply(self, v: int) -> int:
        # OPT: optimize this if needed for compression (likely won't bother benchmarking decompression)
        # (simple vertex-to-vertex mapping; make sure you update it if cycles are added!)
        for cycle in self.cycles:
            try:
                i = cycle.index(v)
                return cycle[(i + 1) % len(cycle)]
            except ValueError:
                pass # not in this cycle
            
        return v # fixed point
    
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
        """make sure the cycles are disjoint and there are no singleons"""

        if any(len(c) < 2 for c in self.cycles):
            raise ValueError("can't have singletons in a permutation")

        all_vertices = set(itertools.chain.from_iterable(self.cycles))
        if len(all_vertices) != sum(map(len, self.cycles)):
            raise ValueError("cycles are not disjoint")


# have to force consistency between encoder and decoder, in case we use different machines (?)
NUMBER_FMT = "<I" # little-endian unsigned int (4 bytes)
SIZEOF_NUM = 4 # bytes

class PermutationEncoding(Enum):
    """different ways to encode a permutation"""
    CYCLES = auto()
    """cycle notation, each (non-trivial) cycle prefixed with its length"""
    PAIRS = auto()
    """pairs of consecutive vertices in cycles
    (no separator needed for decoding, just see when adjacent pairs have no common vertex)"""

    def num_integers_for(self, perm: VertexPermutation):
        """integers in the range [0, n) for a permutation of size n, so they take roughly same space"""
        match self:
            case PermutationEncoding.CYCLES:
                return 1 + perm.nontrivial_cycles + sum(len(cycle) for cycle in perm.cycles)
            case PermutationEncoding.PAIRS:
                # NOTE: do you really need *number* of cycles in the binary too?
                return 1 + 2 * sum(len(cycle) - 1 for cycle in perm.cycles)
            
    @staticmethod
    def default() -> "PermutationEncoding":
        return PermutationEncoding.CYCLES

    def serialize_to(self, file, perm: VertexPermutation) -> None:
        match self:
            case PermutationEncoding.CYCLES:
                file.write(struct.pack(NUMBER_FMT, perm.nontrivial_cycles))
                for cycle in perm.cycles:
                    file.write(struct.pack(NUMBER_FMT, len(cycle)))
                    file.write(struct.pack(NUMBER_FMT, *cycle))
            case PermutationEncoding.PAIRS:
                raise NotImplementedError("not implemented yet")
    
    def deserialize_from(self, file) -> VertexPermutation:
        match self:
            case PermutationEncoding.CYCLES:
                n_cycles = struct.unpack(NUMBER_FMT, file.read(SIZEOF_NUM))[0]
                cycles = []
                for _ in range(n_cycles):
                    cycle_length = struct.unpack(NUMBER_FMT, file.read(SIZEOF_NUM))[0]
                    cycle = list(struct.unpack(NUMBER_FMT * cycle_length, file.read(SIZEOF_NUM * cycle_length)))
                    cycles.append(cycle)
                return VertexPermutation(cycles)
            case PermutationEncoding.PAIRS:
                raise NotImplementedError("not implemented yet")


@dataclass
class SymmetryCompressedGraph:
    residual: nx.Graph
    symmetry: VertexPermutation

    def decompress(self, destructive=False) -> nx.Graph:
        G = self.residual.copy() if not destructive else self.residual
        for e in self.residual.edges:
            e_perm = e
            while True:
                e_perm = self.symmetry(e_perm)
                if eq_edges(e_perm, e):
                    break
                G.add_edge(*e_perm)

        return G

    def remapped_nodes(self, node_mapping: Dict[int, int]) -> "SymmetryCompressedGraph":
        """return version of self with the node labels changed according to mapping"""

        new_resid = nx.relabel_nodes(self.residual, node_mapping, copy=True)
        new_perm = VertexPermutation(cycles=[[node_mapping[v] for v in cycle] for cycle in self.symmetry.cycles])
        return SymmetryCompressedGraph(new_resid, new_perm)
    
    def encoding_size(self, perm_encoding: PermutationEncoding) -> int:
        """as number of integers"""
        return 1 + 2 * self.residual.number_of_edges() + perm_encoding.num_integers_for(self.symmetry)
    
    @property
    def full_size(self) -> int:
        """size of baseline graph encoding, as number of integers"""
        # OPT: store in class attribute at construction, or at least cache this property
        return 2 * self.decompress(destructive=False).number_of_edges()
    
    def relative_efficiency(self, perm_encoding: PermutationEncoding) -> float:
        return (self.full_size - self.encoding_size(perm_encoding)) / self.full_size

    def serialize_to(self, file, perm_encoding: PermutationEncoding) -> None:
        """write the compressed graph to a file"""

        file.write(struct.pack(NUMBER_FMT, self.residual.number_of_edges()))
        for u, v in self.residual.edges:
            file.write(struct.pack(NUMBER_FMT, u, v))

        perm_encoding.serialize_to(file, self.symmetry)


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


def relative_efficiency(sc_graph: SymmetryCompressedGraph, original: nx.Graph, perm_encoding: PermutationEncoding) -> float:
    full_size = 2 * original.number_of_edges()
    zipped_size = 2 * sc_graph.residual.number_of_edges() + perm_encoding.num_integers_for(sc_graph.symmetry)

    return (full_size - zipped_size) / full_size
 

def residual_graph(G: nx.Graph, perm: Permutation) -> Optional[nx.Graph]:
    """Return the residual of G under given vertex symmetry.
    If the given permutation is not a symmetry of G, return None."""
    if set(G.nodes) != set(range(G.number_of_nodes())):
        raise ValueError("graph must have 0-based consecutive integer labels") # otherwise Sympy.Permutation is a pain to use

    G = G.copy()
    repr_edges = []

    while G.number_of_edges() > 0:
        repr_edge = next(iter(G.edges))
        repr_edges.append(repr_edge)
        G.remove_edge(*repr_edge)

        e = repr_edge
        while True:
            e = e[0]^perm, e[1]^perm
            if eq_edges(e, repr_edge):
                break # cycle closed
            
            if G.has_edge(*e):
                G.remove_edge(*e)
            else:
                return None # perm is not a symmetry of G

    return nx.Graph(repr_edges)


def effective_SC_encodings(G: nx.Graph, perm_encoding_strategy: PermutationEncoding) -> Iterator[Tuple[SymmetryCompressedGraph, int]]:
    """yields all SC encodings of G and their absolute efficiency (in num. of integers), when it is positive"""
    if set(G.nodes) != set(range(G.number_of_nodes())):
        raise NotImplementedError("TODO: map to 0-based for searching, then map back")
    
    for p in symmetric(G.number_of_nodes()): # just try every permutation (could probably do better)
        if (resid := residual_graph(G, p)) is None: continue

        scg = SymmetryCompressedGraph(resid, VertexPermutation(p.cyclic_form))
        # NOTE: when assuming 2m integers as baseline, we ignore the extra integer for edgelist *length*
        # (seems reasonable because there is only one for the whole SCPartition,
        #  whereas we *need* to count it for individual SC subgraphs since they add up)
        if (abs_eff := 2 * G.number_of_edges() - scg.encoding_size(perm_encoding_strategy)) > 0:
            yield scg, abs_eff


def best_SC_encoding(G: nx.Graph, perm_encoding_strategy: PermutationEncoding) -> Optional[Tuple[SymmetryCompressedGraph, int]]:
    """return the best SC encoding of G and its absolute efficiency (in num. of integers), if G is compressible"""
    best_eff = 0
    best_enc = None

    for enc, eff in effective_SC_encodings(G, perm_encoding_strategy):
        if eff > best_eff:
            best_eff = eff
            best_enc = enc

    if best_enc is None:
        return None
    else:
        return best_enc, best_eff


def build_graphlet_cache(n: int, perm_encoding: PermutationEncoding, folder="symmetry_cache") -> List[Tuple[SymmetryCompressedGraph, int]]:
    """build a cache of symmetry-compressed graphlets with n vertices.
    best SC encoding chosen based on given permutation encoding strategy"""
    if n > 7:
        raise ValueError("graphlets with more than 7 vertices are not supported")
    
    atlas = nx.graph_atlas_g()
    graphs_on_nodes = {0: 1, 1: 1, 2: 2, 3: 4, 4: 11, 5: 34, 6: 156, 7: 1044}
    start_idx = sum(graphs_on_nodes[i] for i in range(n))

    cache = []

    for i in tqdm(range(start_idx, start_idx + graphs_on_nodes[n]), desc=f"building {n}-vertex graphlet cache (perm-{perm_encoding.name})"):
        if nx.is_connected(G := atlas[i]):
            if (res := best_SC_encoding(G, perm_encoding)) is not None:
                cache.append(res)

    # write to file
    dicts = [{"residual": list(scg.residual.edges), "symmetry": scg.symmetry.cycles, "abs_eff": eff} for scg, eff in cache]
    os.makedirs(folder, exist_ok=True)

    with open(f"{folder}\\SC_graphlets_{n}_perm-{perm_encoding.name}.json", "w") as file:
        json.dump(dicts, file)

    return cache


def load_SC_graphlets(n_max: int, perm_encoding: PermutationEncoding, folder="symmetry_cache") -> List[Tuple[SymmetryCompressedGraph, int]]:
    """load symmetry-compressed graphlets and their abs. efficiencies from the cache (if it exists)"""
    N_MIN = 4 # NOTE: is there a permutation encoding that can compress 3-vertex graphlets?

    res = []
    for n in range(N_MIN, n_max + 1):
        if not os.path.exists(cache_fname := f"{folder}\\SC_graphlets_{n}_perm-{perm_encoding.name}.json"):
            # enc = PermutationEncoding.default()
            print(f"{n}-vertex graphlet cache not found")
            res.extend(build_graphlet_cache(n, perm_encoding=perm_encoding, folder=folder))
        else:
            with open(cache_fname, "r") as file:
                for dict in json.load(file):
                    resid = nx.Graph(dict["residual"])
                    sym = VertexPermutation(dict["symmetry"])
                    res.append((SymmetryCompressedGraph(resid, sym), dict["abs_eff"]))

    return res
        


class SymmetryCompressedPartition:
    """partition of G into its symmetry-compressed subgraphs, + the rest of G"""

    residual: nx.Graph
    """the uncompressed edges of the graph"""
    full_size: int
    """size of the full graph, as number of vertex indices"""
    compressed_graphlets: List[SymmetryCompressedGraph]
    """compressed subgraphs, as a list of graphlet IDs and node mappings"""
    perm_encoding: PermutationEncoding
    """how the permutations are encoded in the resulting binary"""

    def __init__(self, G: nx.Graph, take_ownership=False, perm_encoding=PermutationEncoding.PAIRS): 
        self.residual = G if take_ownership else G.copy()
        self.full_size = 2 * G.number_of_edges()
        self.compressed_graphlets: List[SymmetryCompressedGraph] = []
        self.perm_encoding = perm_encoding

    def compress_subgraph(self, subgraph: nx.Graph, compressed_template: SymmetryCompressedGraph, node_mapping: Dict[int, int]) -> None:
        """replace the subgraph with symmetry-compressed encodin
        - subgraph: the part of the graph to be replaced
        - compressed_template: SC representation of `subgraph`, but with incorrect vertex labels
        - node_mapping: map from template vertices to the vertices of the subgraph
        """
        self.residual.remove_edges_from(subgraph.edges)
        self.compressed_graphlets.append(compressed_template.remapped_nodes(node_mapping))

    @property
    def size(self) -> int:
        """total size of the encoding, as number of integers"""
        return 2 * self.residual.number_of_edges() + sum(scg.encoding_size(self.perm_encoding) for scg in self.compressed_graphlets)
    
    def decompress(self, destructive=False) -> nx.Graph:
        G = self.residual.copy()
        for C in self.compressed_graphlets:
            G.add_edges_from(C.decompress(destructive).edges)

        return G
    
    def serialize_to(self, filename: str) -> None:
        with open(filename, "wb") as file:
            file.write(struct.pack(NUMBER_FMT, self.residual.number_of_edges()))
            for u, v in self.residual.edges:
                file.write(struct.pack(NUMBER_FMT, u, v))

            file.write(struct.pack(NUMBER_FMT, len(self.compressed_graphlets)))
            for g in self.compressed_graphlets:
                g.serialize_to(file, self.perm_encoding)


def graphlet_compress(G: nx.Graph, symmetry_encoding: PermutationEncoding, max_graphlet_sz=7, sort_relative=False, 
                      progress_bar=False) -> SymmetryCompressedPartition:
    """ [ČM21] algorithm 1

    Compress a graph using SC graphlets up to a specified size.
    - G: the graph to be compressed
    - max_graphlet_sz: the maximum size of graphlets to look for
    - sort_relative: if True, sort graphlets by relative efficiency (descending), otherwise absolute efficiency
    """
    graphlets = load_SC_graphlets(n_max=max_graphlet_sz, perm_encoding=symmetry_encoding)

    if sort_relative:
        graphlets.sort(key=lambda scg: scg[0].relative_efficiency(PermutationEncoding.PAIRS), reverse=True)
    else:
        graphlets.sort(key=lambda scg: scg[1], reverse=True)

    Gcomp = SymmetryCompressedPartition(G)
    if progress_bar:
        graphlets = tqdm(graphlets, desc="loop over graphlets")

    for sc_graphlet, _ in graphlets:
        graphlet = sc_graphlet.decompress() # bending over backwards to avoid using a graphlet atlas
        matcher = iso.GraphMatcher(Gcomp.residual, G2=graphlet, node_match=None, edge_match=None) # node and edge attributes are ignored

        for isomorphism in matcher.subgraph_isomorphisms_iter():
            # for subgraph on {0,1,2,3}, an isomorphism looks something like {4: 0, 13: 1, 2: 2, 44: 3}
            inv_iso = {v: k for k, v in isomorphism.items()}
            subgraph = nx.relabel_nodes(graphlet, mapping=inv_iso, copy=True)
            if any(not Gcomp.residual.has_edge(*edge) for edge in subgraph.edges):
                continue # a part of this subgraph has already been compressed (with the same graphlet)
                # NOTE: can this even happen, or does isomorphism iterator adapt on the fly?
            
            Gcomp.compress_subgraph(subgraph, sc_graphlet, inv_iso)

    return Gcomp


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
        """makes a copy of G, and initializes the partition with it"""
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


def bipart_rel_efficiency(g: int, u: int, v: int) -> float:
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
    
    max_eff = bipart_rel_efficiency(Guv.number_of_edges(), len(U), len(V))
    improving = True

    def greedy_optimize(A: Set[int], B: Set[int]) -> None:
        """try to remove a vertex from A to improve efficiency of G(A,B)"""
        nonlocal max_eff, improving
        for a in A:
            Na = list(Guv.neighbors(a)) ;assert all(b in B for b in Na)
            edges_left = Guv.number_of_edges() - len(Na)
            now_isolated = [b for b in Na if Guv.degree[b] == 1] ;assert len(now_isolated) < len(B)

            if (eff := bipart_rel_efficiency(edges_left, len(A) - 1, len(B) - len(now_isolated))) > max_eff:
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
            for entry in subgraph_cache: # OPT: maintain map from edges to cache entries, loop only G(U,V) edges here
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


def foo():
    # print(list(effective_SC_encodings(nx.cycle_graph(4), PermutationEncoding.PAIRS))); quit()
    atlas = nx.graph_atlas_g()
    len2rel_effs = defaultdict(list)

    fig, ax = plt.subplots(nrows=2)
    plt.ion()  # Turn on interactive mode

    for g in atlas[1:]:
        if g.number_of_nodes() > 4: break

        if nx.is_connected(g):
            if (enc := best_SC_encoding(g, PermutationEncoding.PAIRS)) is not None:
                scg, abs_eff = enc
                len2rel_effs[g.number_of_nodes()].append(rel_eff := abs_eff / (2 * g.number_of_edges()))

                # Clear previous plots
                ax[0].clear()
                ax[1].clear()

                # Create new plots
                ax[0].set_title(f"graphlet {g}")
                nx.draw(g, ax=ax[0], with_labels=True)

                ax[1].set_title(f"abs={abs_eff}, rel={rel_eff:.3f} with {scg.symmetry}")
                nx.draw(scg.residual, ax=ax[1],     with_labels=True)

                plt.draw()
                plt.pause(0.1)  # Small pause to ensure plot updates
                
                # Wait for key press to continue
                input("Press Enter to continue...")

    plt.ioff()  # Turn off interactive mode
    plt.close()

    for n, effs in len2rel_effs.items():
        print(f"graphs with {n} nodes: {len(effs)} graphs, avg eff = {sum(effs) / len(effs):.3f}, max eff = {max(effs):.3f}")

    from pprint import pprint
    pprint(len2rel_effs)

if __name__ == "__main__":
    for enc in PermutationEncoding:
        print(f"loading SC graphlets with {enc.name} encoding")
        graphlets = load_SC_graphlets(7, enc)
        abs_effs = [scg[1] for scg in graphlets]
        print(f"mean abs. efficiency: {sum(abs_effs) / len(abs_effs):.3f}, max: {max(abs_effs):.3f}")

    quit()

    # build_graphlet_cache(5, PermutationEncoding.PAIRS)
    graph = net.read_pajek("karate_club", data_folder="data\\networks")
    # graph = nx.erdos_renyi_graph(n=50, p=0.9)
    print(graph)

    Gcomp = graphlet_compress(graph, max_graphlet_sz=7, sort_relative=False, symmetry_encoding=PermutationEncoding.CYCLES)
    print(f"compressed graph size: {Gcomp.size} ({Gcomp.size / 2} pairs)")
