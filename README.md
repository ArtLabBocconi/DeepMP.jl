# DeepMP

A Julia library for message passing on deep neural networks.

## Usage

Use the `solve` function to construct the network, choose the type of message passing in
each layer, and run the message passing iterations:

```julia
using DeepMP

g, W, teacher, E = DeepMP.solve(α=0.2, K=[401,21,3,1]
                   , layers=[:tap,:bp,:bpex]
                   ,r=.9,rstep=0.002, seedx=1,maxiters=800);

@assert E == 0 # zero training error
```
