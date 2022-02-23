multicl = [false]
datasets = [:fashion]
lays = [:tap]
seeds = [2]

K = [0, 501, 501, 0]
ρ = [1.0, 1.0, 1.0]
ψ = [0.99, 0.99, 0.99]
r = [0.9, 0.9, 0.9]
gpu_id = 2

maxiters = 100
ϵ = 1e-4

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