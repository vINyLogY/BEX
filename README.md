# BEX: A general python package to simulate open quantum systems

Seperate graph-related algorithm and tensor operations with the help of GPU accelerated backend for better code structure. 
Still under development and testing, not all test-cases are ready.

## Setup

- Development setup: 
    
    0. Create a python virtural environment with python vesion >= 3.10.

    1. Prepare dependencies: `numpy`, `scipy`, `pytorch`, `torchdiffeq`.

    2. Install `BEX` in develop mode using `pip`:

            pip install -e .

    3. For testing examples, install `tqdm`, `jupyterlab`, `matplotlib`, etc.
