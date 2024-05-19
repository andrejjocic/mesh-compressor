"""network analysis utilities"""

import networkx as nx
from matplotlib import pyplot as plt
from typing import *
import random
import os
from tqdm import tqdm
import re

DEFAULT_DATA_FOLDER = "../networks"


def read_edgelist(filename: str, data_folder=DEFAULT_DATA_FOLDER, progress_bar=False) -> nx.Graph:
    """Reads a network in edgelist (.adj) format. Assumes directed links
    unless the term `undirected` is stated in the header."""
    filename = os.path.splitext(filename)[0]
    
    edges: List[Tuple[int, int]] = []

    with open(os.path.join(data_folder, f"{filename}.adj")) as f:
        comments = ""
        for line in f:
            if line[0] == '#': comments += line[1:]
            else: break

        directed = ("undirected" not in comments)

        if progress_bar and (match := re.search(r"([\d,]+) edges", comments)):
            # get the first edge
            i, j = (int(x) - 1 for x in line.split())
            edges.append((i, j))
            
            m = int(match.group(1).replace(',', ''))
            # get the rest
            for _ in tqdm(range(m - 1), desc=f"reading {filename}"):
                i, j = (int(x) - 1 for x in next(f).split())
                edges.append((i, j))
        else:
            for line in f:
                i, j = (int(x) - 1 for x in line.split())
                edges.append((i, j))

    if directed:
        return nx.DiGraph(edges, name=filename)
    else:
        return nx.Graph(edges, name=filename)
    

def read_pajek(filename: str, data_folder=DEFAULT_DATA_FOLDER,
               label_parser: Callable[[str, str], Dict[str, Any]] = None) -> nx.Graph:
    """Reads a graph in Pajek (.net) format with at most one value
    attached to each node (aside from the label). Note that this doesn't entirely
    comply with the Pajek format specification, see
    http://vlado.fmf.uni-lj.si/pub/networks/pajek/doc/draweps.htm

    - label_parser: a function that takes a node's label and value (default None),
    and returns a dictionary of node attributes to be stored in graph. By default,
    labels will be stored in attribute 'label', and values (if present) in 'value'."""
    
    filename = os.path.splitext(filename)[0]

    if label_parser is None:
        def label_parser(lab, val): return \
            {"label": lab} if val is None else {"label": lab, "value": val}

    with open(os.path.join(data_folder,  f"{filename}.net"), encoding="UTF-8") as file:
        file.readline() # skip header
        nodes = [] # OPT pre-allocate given header

        for line in file:
            if line.startswith("%"): continue # skip comments
            if line.startswith("*"):
                match line.split()[0][1:]: # TODO extract m for optional progressbar
                    case "edges": G = nx.MultiGraph(name=filename)
                    case "arcs": G = nx.MultiDiGraph(name=filename)
                    case link_type: raise SyntaxError(f"invalid link type: {link_type}")
                break
            else: # add node
                match line.strip().split("\""):
                    case num, lab:
                        nodes.append((int(num) - 1, label_parser(lab, None)))
                    case num, lab, val:
                        nodes.append((int(num) - 1, label_parser(lab, val)))
                    case _:
                        raise SyntaxError("failed to parse " + line)

        G.add_nodes_from(nodes)

        for line in file:
            i, j = (int(x) - 1 for x in line.split()[:2])
            G.add_edge(i, j)

    return G


def lcc(G: nx.Graph) -> float:
    """relative size of the largest connected component (between 0 and 1)"""
    if G.is_directed(): G = nx.Graph(G)

    return len(max(nx.connected_components(G), key=len)) / len(G)


def info(G: nx.Graph, clustering_sample: int | None = 10_000) -> Dict[str,str]:
    """Prints and returns basic information on the provided graph.
    - If clustering_sample is given, average clustering will be computed from
    a sample of nodes (of given size).
    """
    stats = {str(G).split()[0]: f'"{G.name}"'}

    n = G.number_of_nodes()
    m = G.number_of_edges()

    stats["Nodes"] = f"{n:,d} (iso={nx.number_of_isolates(G)})"
    stats["Edges"] = f"{m:,d} (loop={nx.number_of_selfloops(G)})"

    if G.is_directed():
        stats["Degree"] = f"{m / n:.2f}"
        for spec, degree_map in [("in", G.in_degree()), ("out", G.out_degree())]:
            degrees = [k for _, k in degree_map]
            stats["Degree"] += f" {spec}=[{min(degrees):,d}..{max(degrees):,d}]"
        
        stats["Density"] = f"{m / (n*(n - 1)):.2f}"
    else:
        degrees = [k for _, k in G.degree()]
        stats["Degree"] = f"{2 * m / n:.2f} [{min(degrees):,d}..{max(degrees):,d}]"
        stats["Density"] = f"{2*m / (n*(n - 1)):.2f}"

    C = list(nx.connected_components(nx.Graph(G) if G.is_directed else G))
    largest_comp = max(C, key=len)

    stats["LCC"] = f"{100 * len(largest_comp) / n:.1f}% (n={len(C):,d})"

    if clustering_sample is not None:
        if isinstance(G, nx.MultiGraph):
            G = nx.Graph(G)

        clustering_on = G.nodes if n <= clustering_sample \
            else random.sample(list(G.nodes), k=clustering_sample)

        stats["Clustering"] = f"{nx.average_clustering(G, clustering_on):.4f}"

    for name, value in stats.items():
       print(f"{name:>12s} | {value}")
    print()
    return stats


def plot_degree(G: nx.Graph, save_path: str | None = None) -> None:
    """Plots degree distribution(s) on a log-log plot.
    If save_path is given, the plot will be saved in given folder/file
    instead of being shown."""
    plt.clf()
    plt.title(G.name + " degree distribution")
    plt.ylabel('$p_k$')
    plt.xlabel('$k$')
    n = G.number_of_nodes()

    def aux(degree_view: Iterable, k_min: int, **kwargs):
        degree_count = Counter(k for _, k in degree_view)
        k_max = max(degree_count.keys())

        x = list(range(k_min, k_max + 1))
        y = [degree_count[i] / n for i in x]
        plt.loglog(x, y, 'o', **kwargs)

    if G.is_directed():
        aux(G.out_degree, k_min=0, label="outdegree", color="pink")
        aux(G.in_degree, k_min=0, label="indegree", color="purple", alpha=0.5)
        plt.legend()
    else:
        aux(G.degree, k_min=1, color="gray")

    if save_path is None:
        plt.show()
    else:
        if len(os.path.splitext(save_path)[1]) == 0: # no extension
            plt.savefig(os.path.join(save_path, f"{G.name}_degree_distro.pdf"))
        else:
            plt.savefig(save_path)


def draw_graph(G: nx.Graph, title="", **kwargs):
    plt.title(f"{G.name} {title}")
    nx.draw(G, with_labels=True, **kwargs)
    plt.show()


def find_node(G: nx.Graph, label: str):
    """Finds node with given label in G."""
    for i, data in G.nodes(data=True):
        if data['label'] == label:
            return i
    raise ValueError(f"node '{label}' not found in {G.name}")



def ER_random_graph(n: int, m: int) -> nx.MultiGraph:
    """Returns Erdős–Rényi random graph with n nodes and m edges."""
    G = nx.MultiGraph(name="ER")
    for i in range(n):
        G.add_node(i)

    edges = []
    while m > 0:
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i != j:
            # G.add_edge(i, j)
            edges.append((i, j))
            m -= 1

    G.add_edges_from(edges)  # avoids checking presence of edges
    return G


