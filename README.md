# CPU-side Simulator for MSCCL XML Files


This repo contains the code that simulates a collective communication algorithm written in XML with CPU threads.

To build this repo, use the following instructions.
```shell
mkdir -p build && cd build
cmake .. && make
```

Currently, we provide the following verifiers:
1. An `allgather-verifier` that verifies the validity of an algorithm written for out-of-place AllGather.
2. An `alltoall-verifier` that verifies the validity of an algorithm written for out-of-place AllToAll with uniform buffer parition.
3. An `alltoallv-verifier` that verifies the validity of an algorithm written for out-of-place AllToAll with variable buffer parition.

To run a verification, use `./<verifier> <xml> <run_iters>`.
It will execute the algorithm for the specified number of times (`run_iters`) and check whether the output buffer is correct.
Note that `alltoallv-verifier` takes an additional input csv file `./alltoallv-verifier <xml> <run_iters> <csv>`.
This file should contain $W\times W$ integer values, given that $W$ is the world size (i.e., `ngpus` in the XML file).
The cell at the $i$-th row and $j$-th column means the number of chunks that are sent from rank $i$ to rank $j$.

# Key Idea of Simulation
We simulate a GPU threadblock with a CPU thread, because instructions within a threadblock are executed sequentially.
We simulate neighbouring peers in a channel via a FIFO queue (called `Mailbox` in the source file).

Class organization is as follows.
A `CommGroup` internally holds all of its `GpuRank`s.
A `GpuRank` internally holds all of its `ThreadBlock`s, as well as the input/output/scratch buffers.

There is NO lock protecting these buffers, although they may be concurrently read or written by multiple threads.
Note that any data hazard should be avoided by specifying correct dependencies in the XML file.

In each run, all `ThreadBlock`s in all `GpuRank`s will execute in parallel.
The channels are built only once prior to the start of the first run, similar to channels in MSCCL and NCCL.
