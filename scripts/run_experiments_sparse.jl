multicl = [true]
datasets = [:mnist]
lays = [:bpi]
seeds = [2]
gpu_id = 1
density = 0.1
#density = 1.0

for multiclass in multicl, dataset in datasets, lay_type in lays, seed in seeds
    ρ = [1.0, 1.0, 0.9]
    #try
        run_experiment(; multiclass, dataset, lay_type, seed, ρ, density,
        epochs=100, batchsize=128, usecuda=true, gpu_id, ψ=[0.79, 0.79, 0.79], 
        M=Int(6e4), maxiters=1, r=0.0, ϵinit=1.0, K=[0, 101, 101, 0], 
        altsolv=false, altconv=false, saveres=true);
    #catch
    #    println("a process has been interrupted")
    #end
end