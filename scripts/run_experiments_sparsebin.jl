multicl = [false]
datasets = [:mnist]
lays = [:bpi]
seeds = [2]
gpu_id = 1
density = 0.1

K = [0, 101, 101, 0]
ρ = [1.0, 1.0, 0.9]
ψ = [0.8, 0.8, 0.8]

for multiclass in multicl, dataset in datasets, lay_type in lays, seed in seeds
    #try
        run_experiment(; multiclass, dataset, lay_type, seed, ρ, density,
        epochs=100, batchsize=128, usecuda=true, gpu_id, ψ, 
        M=Int(6e4), maxiters=1, r=0.0, ϵinit=1.0, K, 
        altsolv=false, altconv=false, saveres=true);
    #catch
    #    println("a process has been interrupted")
    #end
end