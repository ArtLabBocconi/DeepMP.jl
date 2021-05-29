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

## Experiments

A typical experiment we run is 

```julia
include("scripts/real_data_experiments.jl")

run_experiment(9; usecuda=true, gpu_id=0, epochs=10, lay=:tap, batchsize=128, 
                ρ=1+1e-4, ψ=0.8, M=-1,dataset=:fashion, maxiters=1, r=0., ϵinit=1., 
                K=[28*28,101,101,1], 
                altsolv=true, altconv=true)
```
