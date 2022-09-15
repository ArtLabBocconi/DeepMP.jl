multicl = [true]
datasets = [:fashion]
lays = [:tap]
seeds = [2]

maxiters = 10000
ϵ = 1e-5

K = [0, 501, 501, 0]
ρ = [1.0, 1.0, 1.0]
ψ = [0.999, 0.999, 0.999]
r = [0.0, 0.0, 0.0]
gpu_id = 1

for multiclass in multicl, dataset in datasets, lay_type in lays, seed in seeds
    #try
        run_experiment(; multiclass, dataset, lay_type, seed, ρ,
        epochs=1, batchsize=-1, usecuda=true, gpu_id, 
        ψ, M=Int(6e4), maxiters, r,
        ϵinit=1.0, K, ϵ,
        altsolv=false, altconv=false, saveres=true, verbose=2);
    #catch
    #    println("a process has been interrupted")
    #end
end