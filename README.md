# python-orca
Orbit counting algorithm implemented by python.

## Description
**Orca** is an efficient algorithm for counting graphlets in networks. 
It computes node- and edge-orbits (of 4- and 5-node graphlets) for each node in the network.

To improve the understanding of the orca for beginners, we re-implemented orca algorithm
using python.

To improve the efficiency of python code, we use python-numba-cuda to accelerate orca on GPU.

## Installation
We have three requirement packages: cupy, cudf, and numba. 
```bash
conda create -n cupy-orca -c rapidsai -c nvidia -c conda-forge cudf=22.08 python=3.8 cudatoolkit=11.5
conda deactivate
conda activate cupy-orca
```
