# DeepMP

A Julia library for message passing on deep neural networks.

## Usage

Use the `solve` function to construct the network, choose the type of message passing in
each layer, and run the message passing iterations:

<!-- 
```julia
using DeepMP, Random

Random.seed!(17)

prob = DeepMP.generate_problem(N=401, Mtrain=1000)

g, W, teacher, E = DeepMP.solve(prob.xtrain, prob.ytrain; 
                        K = [401, 101, 1], 
                        layers=[:bp, :bp], 
                        ψ=[0.8, 0.8], ϵinit=1.0 , r=.9, rstep=0.002, 
                        maxiters=800, 
                        batchsize=128,
                        usecuda=true, gpu_id=0);

@assert E == 0 # zero training error
```
 -->

## Experiments

A typical experiment we run is 

```julia
include("scripts/real_data_experiments.jl")

run_experiment(; multiclass=false, dataset=:fashion, lay_type=:bp, seed=2, 
            ρ=[1.0, 1.0, 0.9], density=1.0, ψ=[0.8, 0.8, 0.8],
            epochs=200, batchsize=128, usecuda=true, gpu_id=0,  
            M=Int(6e4), maxiters=1, r=0.0, ϵinit=1.0, K=[28*28, 501, 501, 1], 
            altsolv=false, altconv=false, saveres=false);
```
