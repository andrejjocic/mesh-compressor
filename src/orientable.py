from typing import *
from collections import defaultdict
import itertools


Vertex = int
Edge = Tuple[Vertex, Vertex]
Triangle = Tuple[Vertex, Vertex, Vertex]
Simplex = Tuple[Vertex, ...]

class SimplexMap[V]:
    """A map from simplices to values of type V (with key order invariance)"""
    _map: Dict[Simplex, V]
    _input_dimension: Optional[int]

    def __init__(self, key_dimension: Optional[int], map_constructor: Callable[[], Dict[Simplex, V]] = dict):
        """key_dimension=None for no restriction on simplex dimension"""
        self._map = map_constructor()
        self._input_dimension = key_dimension

    def __getitem__(self, simplex: Simplex) -> V:
        if self._input_dimension is not None:
            assert len(simplex) == self._input_dimension + 1
        return self._map[tuple(sorted(simplex))]
    
    def __setitem__(self, simplex: Simplex, value: V):
        if self._input_dimension is not None:
            assert len(simplex) == self._input_dimension + 1
        self._map[tuple(sorted(simplex))] = value

    def values(self) -> Iterator[V]:
        return self._map.values()
    
    def __contains__(self, simplex: Simplex) -> bool:
        return tuple(sorted(simplex)) in self._map
    
    def __iter__(self) -> Iterator[Simplex]:
        return iter(self._map)
    
    def items(self) -> Iterator[Tuple[Simplex, V]]:
        return self._map.items()



def orientableQ(T: List[Triangle]) -> bool:
    """T: triangulation of a 2-manifold"""
    return orient(T) is not None


def edges(triangle: Triangle) -> Iterator[Edge]:
    yield from itertools.combinations(triangle, r=2)

def neighbor_map(triangulation: List[Triangle], combinatorial_surface=True) -> SimplexMap[List[Triangle]]:
    """Return a map from triangles to their neighbors (by shared edge)"""
    triangs_on: SimplexMap[List[Triangle]] = SimplexMap(1, map_constructor=lambda: defaultdict(list))
    for tri in triangulation:
        for edge in edges(tri): triangs_on[edge].append(tri)

    # remove duplicate triangles (happens for projective plane)
    for edge, tris in triangs_on.items():
        triangs_on[edge] = list(set(tuple(sorted(t)) for t in tris))

    if combinatorial_surface:
        try:
            # no edge in a combinatorial surface can be shared by more than two triangles
            edge, tris = next((edge, tris) for edge, tris in triangs_on.items() if not 1 <= len(tris) <= 2)
            raise Exception(f"Edge {edge} is shared by {len(tris)} triangles: {tris}")
        except StopIteration:
            pass

    neighbors_of: SimplexMap[List[Triangle]] = SimplexMap(2, map_constructor=lambda: defaultdict(list))
    for triangle in triangulation:
        for edge in edges(triangle):
            for neighbor in triangs_on[edge]:
                if sorted(neighbor) != sorted(triangle): neighbors_of[triangle].append(neighbor)

    return neighbors_of


def orient(T: List[Triangle], reverse_first=False) -> Optional[List[Triangle]]:
    """Orient the faces of a triangulation (if possible)
    ### Args:
    - T: triangulation of a 2-manifold (locally plane-like, optional boundary)
    - reverse_first: if True, reverse the orientation of the first triangle before fixing the rest
    ### Returns:
    None if T is not orientable, otherwise a list of oriented triangles
    """
    neighbors_of = neighbor_map(T)
    triang_orientation: SimplexMap[Triangle] = SimplexMap(2)

    first = T[0]
    triang_orientation[first] = first[::-1] if reverse_first else first # arbitrary orientation, defines the rest
    orientation_queue = list(neighbors_of[first])
    
    while orientation_queue: # NOTE: BFS faster at finding a contradiction?
        if (tri := orientation_queue.pop()) in triang_orientation:
            continue # already reached from another neighbor (and oriented consistently)

        # A 2-manifold is orientable if you can choose the orientations
        # of all its triangles consistently. Two triangles that share
        # an edge are consisently oriented if they induce opposite orientations on the common edge.
        
        existing_edge_orientations: List[Edge] = []
        for neighbor in neighbors_of[tri]:
            if neighbor in triang_orientation:
                existing_edge_orientations.append(
                    induced_orientation(shared_edge(tri, neighbor), triang_orientation[neighbor]))

        # because we are doing DFS, we know at least one edge is oriented
        match existing_edge_orientations:
            case [o]: # orient this triangle consistently with the 1 oriented edge
                a, b = o
                c = next(v for v in tri if v not in o)
                triang_orientation[tri] = (b, a, c)
        
            # if there are 2 or 3 oriented edges, we must:
            # - check that the existing orientations match (for this triangle); otherwise return None
            # - orient this triangle consistently with the existing orientations
            case [o1, o2]:
                if (induced_tri := combine_edge_orietations(o1, o2)) is None:
                    return None
                else:
                    triang_orientation[tri] = induced_tri[::-1]

            case [o1, o2, o3]:
                # check same condition as above, but for two pairs of edges (all 3 is redundant)
                if (o12 := combine_edge_orietations(o1, o2)) is None: return None
                if (o23 := combine_edge_orietations(o2, o3)) is None: return None
                
                if cyclically_equal(o12, o23):
                    triang_orientation[tri] = o12[::-1]
                else:
                    return None                
            case _:
                raise Exception(f"triangle {tri} has {len(existing_edge_orientations)} oriented edges")
            
        orientation_queue.extend(neighbors_of[tri])
            
    return list(triang_orientation.values())


def shared_edge(tri1: Triangle, tri2: Triangle) -> Edge:
    """Return the edge shared by two triangles"""
    common_vertices = set(tri1) & set(tri2)
    assert len(common_vertices) == 2
    return tuple(common_vertices)


def induced_orientation(edge: Edge, triangle: Triangle) -> Edge:
    """Return the orientation of an edge induced by the orientation of a triangle"""
    assert set(edge) <= set(triangle)
    u, v = edge
    i = triangle.index(u)
    j = triangle.index(v)

    if (i + 1) % 3 == j:
        return u, v
    else:
        return v, u

def combine_edge_orietations(o1: Edge, o2: Edge) -> Triangle:
    """Combine two edge orientations into a triangle orientation (if they don't conflict)""" 
    common_vtx = next(iter(set(o1) & set(o2)))
    if o1.index(common_vtx) == o2.index(common_vtx):
        return None # orientations conflict
    
    only_in_o2 = next(v for v in o2 if v not in o1)
    return o1 + (only_in_o2,)

def cyclically_equal(s1: Simplex, s2: Simplex) -> bool:
    """Check if two simplices are equal up to cyclic permutation of vertices"""
    assert len(s1) == len(s2), "Simplex dimensions must match"
    return any(s1 == s2[i:] + s2[:i] for i in range(len(s2)))