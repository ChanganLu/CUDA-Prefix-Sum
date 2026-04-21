# CUDA-Prefix-Sum

This repository is an implementation of CUDA prefix sum.

The key idea is to launch the kernel function only once to avoid the overhead of launching the kernel, while using a fixed number of CUDA blocks to simulate the processing flow of multiple logical blocks, and employing a lookback strategy.
