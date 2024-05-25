# Polygonal Model Compression with Graph Symmetries

A tool for compressing polygonal models (in PLY format). Compresses the connectivity data by exploiting symmetries [1].
Implemented as part of the Advanced Computer Graphics course at UL FRI.

### Instructions
Running `python src/mesh_utils.py --help` prints instructions for compressing and decompressing a mesh.

### Project structure
The main source files in `/src` are:
- `mesh_utils.py`: utilities for working with PLY files, including (de-)compression
- `graphlets.py`: compression of undirected graphs, based on a graphlet dictionary
- `symmetry_compressor.py`: implementation of an algorithm from [1], not fully integrated with the `mesh_utils` interface (due to unpromising initial results)


[1] U. Čibej, J. Mihelič, ["Graph automorphisms for compression,"](https://www.degruyter.com/document/doi/10.1515/comp-2020-0186/html) Open Computer Science, vol. 11, no. 1, pp. 51-59, 2020, doi: 10.1515/comp-2020-0186.