multicl = [true]
datasets = [:fashion]
lays = [:bpi, :tap, :mf]
seeds = [2, 7, 11]
gpu_id = 1

K = [0, 2048, 2048, 2048, 0]
ψ = [0.8, 0.8, 0.8, 0.8]

density = 1.0

for multiclass in multicl, dataset in datasets, lay_type in lays, seed in seeds
    ρ = [1.0, 1.0, 1.0, 0.9]
    #try
        run_experiment(; multiclass, dataset, lay_type, seed, ρ, density,
        epochs=200, batchsize=128, usecuda=true, gpu_id, ψ, 
        M=Int(6e4), maxiters=1, r=0.0, ϵinit=1.0, K, 
        altsolv=false, altconv=false, saveres=true);
    #catch
    #    println("a process has been interrupted")
    #end
end