import unittest
import networkx as nx
from collections import defaultdict
from pprint import pprint
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from symmetry_compressor import *
import network_utils as net


class TestCompressor(unittest.TestCase):

    def assertEqualGraphs(self, g1: nx.Graph, g2: nx.Graph):
        self.assertEqual(g1.number_of_nodes(), g2.number_of_nodes())
        self.assertEqual(g1.number_of_edges(), g2.number_of_edges())
        # no need to complicate with isomorphism test, decompressed graph
        # should have the same nodes and edges (provided no isolated nodes?)
        self.assertCountEqual(g1.nodes, g2.nodes)
        self.assertCountEqual(g1.edges, g2.edges)

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
    

    def test_karate(self):
        karate = nx.Graph(net.read_pajek("karate_club", data_folder="data\\networks"))
        assert len(next(iter(karate.edges))) == 2 # edge labels may break something
        C = compress_bipartite(karate)
        self.assertEqual(C.size, 55, msg="maybe due to different vertex ordering?") # [ČM21] table 3
        karate_dec = C.decompress()
        self.assertEqualGraphs(karate, karate_dec)

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

    # def test_cliques_bipartite(self):
    #     G = nx.complete_bipartite_graph(3, 4)
    #     self.test_bipartite_compression(G)

    # def test_cliques_random(self):
    #     G = nx.random_regular_graph(3, 10)
    #     self.test_bipartite_compression(G)

    # def test_cliques_er(self):
    #     G = nx.erdos_renyi_graph(n=5, p=0.75)
    #     # nx.draw(G, with_labels=True); plt.show()
    #     self.test_bipartite_compression(G)
    


if __name__ == "__main__":
    unittest.main()