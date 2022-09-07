# python-orca
Orbit counting algorithm implemented by python.

## Description
**Orca** is an efficient algorithm for counting graphlets in undirect and no-selfloop networks. 
It computes node- and edge-orbits (of 4- and 5-node graphlets) for each node in the network.

To improve the understanding of the orca for beginners, we re-implemented orca algorithm
using python.

To improve the efficiency of python code, we use python-numba-cuda to accelerate orca on GPU.

## Installation
To accelerate orca on GPU, we have three requirement packages: cupy, cudf, and numba. 
```bash
conda create -n cupy-orca -c rapidsai -c nvidia -c conda-forge cudf=22.08 python=3.8 cudatoolkit=11.5
conda deactivate
conda activate cupy-orca
pip install numba==0.55.2
```

## Usage
The utility takes four command-line arguments:
`python orca.py --orbit-type node --graphlet-size 5 --input-file example_graph.csv --output-file orbit-counts.csv`

1. *Orbit type* is either `node` or `edge`.
2. *Graphlet size* indicates the size of graphlets that you wish to count and should be either `4` or `5`.
3. *Input file* describes the network in a simple text format. The first line contains two integers n and e - the number of nodes and edges. The following e lines describe undirected edges with space-separated ids of their endpoints. Node ids should be between 0 and n-1. See example_graph.csv as an example.
4. *Output file* will consist of n lines, one for each node in a graph from 0 to n-1. Every line will contain space-separated orbit counts depending on the specified graphlet size and orbit type.

## Data

The random Erdos-Renyi graph used in the experiment is available in the `example_graph.zip` archive.
