import unittest
import networkx as nx
from collections import defaultdict
from pprint import pprint
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from symmetry_compressor import *
import network_utils as net
import random
import graphlets
from pathlib import Path


class TestAtlasCompressor(unittest.TestCase):

    maxDiff = None
    
    def assertEqualGraphs(self, g1: nx.Graph, g2: nx.Graph): 
        self.assertEqual(g1.number_of_nodes(), g2.number_of_nodes())
        self.assertEqual(g1.number_of_edges(), g2.number_of_edges())
        # no need to complicate with isomorphism test, decompressed graph
        # should have the same nodes and edges (provided no isolated nodes?)
        self.assertCountEqual(g1.nodes, g2.nodes)
        self.assertCountEqual(map(sorted, g1.edges), map(sorted, g2.edges)) # disregard edge order (undirected)

    def assertValidEfficiency(self, eff: float):
        self.assertGreaterEqual(eff, 0)
        self.assertLessEqual(eff, 1)

    # TODO figure out how to put above helpers in a base class

    def test_karate(self):
        karate = nx.Graph(net.read_pajek("karate_club", data_folder="data\\networks"))
        comp = graphlets.compress_subgraphlets(karate, max_graphlet_sz=5)
        self.assertValidEfficiency(comp.relative_efficiency)
        karate_dec = comp.decompress()
        self.assertEqualGraphs(karate, karate_dec)

    def validate_serialization(self, G: nx.Graph, **compressor_kwargs):
        C = graphlets.compress_subgraphlets(G, **compressor_kwargs)
        self.assertValidEfficiency(C.relative_efficiency)
        bin_path = Path("test.acgf") # atlas-compressed graph file
        C.serialize(bin_path)
        C2, _ = graphlets.AtlasCompressedGraph.deserialize(bin_path)
        bin_path.unlink()
        G2 = C2.decompress()
        self.assertEqualGraphs(G, G2)

    def test_serialization_karate(self):
        karate = nx.Graph(net.read_pajek("karate_club", data_folder="data\\networks"))
        self.validate_serialization(karate, max_graphlet_sz=6)

    def test_serialization_random(self):
        G = nx.erdos_renyi_graph(n=20, p=0.75)
        self.validate_serialization(G, max_graphlet_sz=4)
    

class TestSymmetryCompressor(unittest.TestCase):

    def assertEqualGraphs(self, g1: nx.Graph, g2: nx.Graph): 
        self.assertEqual(g1.number_of_nodes(), g2.number_of_nodes())
        self.assertEqual(g1.number_of_edges(), g2.number_of_edges())
        # no need to complicate with isomorphism test, decompressed graph
        # should have the same nodes and edges (provided no isolated nodes?)
        self.assertCountEqual(g1.nodes, g2.nodes)
        self.assertCountEqual(map(sorted, g1.edges), map(sorted, g2.edges)) # disregard edge order (undirected)


    def test_decomp_4cycle(self):
        # [ČM21] table 1
        G = nx.cycle_graph(4)
        G = nx.convert_node_labels_to_integers(G, first_label=1)
        G_resid = G.copy()
        G_resid.remove_edge(2, 3)
        G_decomp = decompress_SC(G_resid, VertexPermutation([[1,2], [4,3]]))
        self.assertEqualGraphs(G, G_decomp)

        G_resid.remove_edge(4, 3)
        G_decomp = decompress_SC(G_resid, VertexPermutation([[1,3]]))
        self.assertEqualGraphs(G, G_decomp)
        
        G_resid.remove_edge(1, 4)
        G_decomp = decompress_SC(G_resid, VertexPermutation([[1,2,3,4]]))
        self.assertEqualGraphs(G, G_decomp)


    def test_decomp_NSC_example(self):
        # [ČM21] fig. 2
        Hpi = nx.Graph()
        Hpi.add_edges_from((int(e[0]), int(e[1])) for e in "12 14 15 16 17 23 26 27 67".split())
        self.assertEqual(Hpi.number_of_edges(), 9)

        comp = NSCompressedGraph(
            G_diff_H=[(6,3)], # will be removed from H to make G
            H_resid=Hpi,
            perm=VertexPermutation([[1,4], [2,3]])
        )
        self.assertEqual(comp.size, 2 + 9 + 1)
        self.assertEqual(comp.decompress().number_of_edges(), 14)

    # def test_karate_graphlets(self):
    #     karate = nx.Graph(net.read_pajek("karate_club", data_folder="data\\networks"))
    #     comp = graphlet_compress(karate, symmetry_encoding=PermutationEncoding.PAIRS, max_graphlet_sz=5) # paper shows 5 is max pattern
    #     karate_dec = comp.decompress()
    #     self.assertEqualGraphs(karate, karate_dec)

    def validate_serialization(self, G: nx.Graph, **compressor_kwargs):
        C = graphlet_compress(G, symmetry_encoding=PermutationEncoding.CYCLES, **compressor_kwargs)
        bin_path = Path("test.bin") # compressed graph file
        C.serialize_to(bin_path)
        C2, _ = SymmetryCompressedPartition.deserialize_from(bin_path, PermutationEncoding.CYCLES)
        bin_path.unlink()
        G2 = C2.decompress()
        self.assertEqualGraphs(G, G2)

    def test_serialization_karate(self):
        karate = nx.Graph(net.read_pajek("karate_club", data_folder="..\\data\\networks"))
        assert len(next(iter(karate.edges))) == 2 # edge labels may break something``
        self.validate_serialization(karate, progress_bar=False, cache_folder="..\\symmetry_cache")

    def test_karate_bipart(self):
        karate = nx.Graph(net.read_pajek("karate_club", data_folder="data\\networks"))
        assert len(next(iter(karate.edges))) == 2 # edge labels may break something
        C = compress_bipartite(karate, caching_mode=CachingMode.NONE)
        self.assertEqual(C.size, 55, msg="maybe due to different vertex ordering?") # [ČM21] table 3
        karate_dec = C.decompress()
        self.assertEqualGraphs(karate, karate_dec)
        
        C = compress_bipartite(karate, caching_mode=CachingMode.DYNAMIC)
        self.assertLessEqual(C.relative_efficiency, 1)
        self.assertEqual(C.size, 55, msg="maybe due to different vertex ordering?")
        karate_dec = C.decompress()
        self.assertEqualGraphs(karate, karate_dec)        
        
        Cstat = compress_bipartite(karate, caching_mode=CachingMode.STATIC)
        self.assertLessEqual(Cstat.relative_efficiency, C.relative_efficiency)
        self.assertLessEqual(0, Cstat.relative_efficiency)
        self.assertEqualGraphs(karate, Cstat.decompress())   

    def assertValidEfficiency(self, eff: float):
        self.assertGreaterEqual(eff, 0)
        self.assertLessEqual(eff, 1)

    def validateBipartiteCompression(self, G: nx.Graph, try_cacheless=False):
        if try_cacheless:
            C = compress_bipartite(G, caching_mode=CachingMode.NONE)
            rel_eff = C.relative_efficiency
            self.assertValidEfficiency(rel_eff)
            self.assertEqualGraphs(G, C.decompress())
        else:
            rel_eff = None

        C = compress_bipartite(G, caching_mode=CachingMode.DYNAMIC)
        self.assertValidEfficiency(C.relative_efficiency)
        if rel_eff is not None:
            self.assertAlmostEqual(C.relative_efficiency, rel_eff, places=1, msg="caching changes efficiency (different tie-breaks?)")

        self.assertEqualGraphs(G, C.decompress())

        Cstat = compress_bipartite(G, caching_mode=CachingMode.STATIC)
        self.assertValidEfficiency(Cstat.relative_efficiency)
        self.assertLessEqual(Cstat.relative_efficiency, C.relative_efficiency)
        self.assertEqualGraphs(G, Cstat.decompress())

    def test_bipart_small_er(self):
        G = nx.erdos_renyi_graph(n=20, p=0.75)
        self.validateBipartiteCompression(G, try_cacheless=True)
    
    def test_bipart_big_er(self):
        G = nx.erdos_renyi_graph(n=40, p=0.5)
        self.validateBipartiteCompression(G, try_cacheless=False)
    
    def test_complete_bipartite(self):
        G = nx.complete_bipartite_graph(3, 4)
        self.validateBipartiteCompression(G, try_cacheless=True)

    def test_noncomplete_bipartite(self):
        u, v = 3, 5
        G = nx.complete_bipartite_graph(u, v)
        for _ in range(2):
            G.remove_edge(random.randint(0, u-1), random.randint(u, u+v-1))
        self.validateBipartiteCompression(G, try_cacheless=True)


    # TODO: assert some things from paper theorems (trees not compressible...)

    # def test_cliques_complete(self):
    #     G = nx.complete_graph(5)
    #     self.test_bipartite_compression(G)

    # def test_cliques_cycle(self):
    #     G = nx.cycle_graph(6)
    #     self.test_bipartite_compression(G)

    # def test_cliques_star(self):
    #     G = nx.star_graph(5)
    #     self.test_bipartite_compression(G)

    # def test_cliques_random(self):
    #     G = nx.random_regular_graph(3, 10)
    #     self.test_bipartite_compression(G)



if __name__ == "__main__":
    unittest.main()