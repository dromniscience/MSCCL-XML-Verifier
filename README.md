# CPU-side Simulator for MSCCL XML Files


This repo contains the code that simulates a collective communication algorithm written in XML with CPU threads.

To build this repo, use the following instructions.
```shell
mkdir -p build && cd build
cmake .. && make
```

Currently, we provide an `allgather-verifier` that verifies the validity of an algorithm written for out-of-place AllGather.

To run a verification, use `./allgather-verifier <xml> <run_iters>`.
It will execute the algorithm for the specified number of times and check whether the output buffer is correct.

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
