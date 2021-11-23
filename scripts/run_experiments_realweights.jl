#includet("real_data_experiments.jl")

multicl = [false]
datasets = [:mnist]
lays = [:cbpi]
seeds = [2, 7, 11, 15, 19]
gpu_id = 0
density = 1.0
usecuda = false

K = [0, 101, 101, 0]
ρ = [1.0, 1.0, 0.9]
#ρ = [1.001, 1.001, 1.0]
ψ = [0.8, 0.8, 0.8]

for multiclass in multicl, dataset in datasets, lay_type in lays, seed in seeds
    #try
        run_experiment(; multiclass, dataset, lay_type, seed, ρ, density,
            epochs=100, batchsize=128, usecuda, gpu_id, ψ, 
            M=Int(6e4), maxiters=1, r=0.0, ϵinit=1.0, K, 
            altsolv=false, altconv=false, saveres=true);
    #catch
    #    println("a process has been interrupted")
    #end
end