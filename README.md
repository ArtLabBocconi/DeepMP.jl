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

run_experiment(; dataset=:fashion, multiclass=false,
                 usecuda=true, gpu_id=0, epochs=100, 
                 layers=[:bpi, :bpi, :bpi], 
                 batchsize=128, 
                 ρ=[1.0+1e-6, 1.0+1e-6, 0.0], 
                 ψ=0.2, M=60000, 
                 maxiters=1, r=0.0, 
                 ϵinit=1.0, K=[28*28, 101, 101, 1], altsolv=false, altconv=true, seed=2, saveres=false)
```
