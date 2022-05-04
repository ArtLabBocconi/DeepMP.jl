multicl = [true]
datasets = [:fashion]
lays = [:bpi]
seeds = [2]
gpu_id = 0

K = [0, 512, 512, 0]
ρ = [1.0, 1.0, 0.9]
ψ = [0.8, 0.8, 0.8]

epochs = 10
batchsize = 128
ϵinit = 1.0
r = 0.0

density = 1.0

for multiclass in multicl, dataset in datasets, lay_type in lays, seed in seeds
    #try
        run_experiment(; multiclass, dataset, lay_type, seed, ρ, density,
        epochs, batchsize, usecuda=true, gpu_id, ψ, 
        M=Int(6e4), maxiters=1, r, ϵinit, K, 
        altsolv=false, altconv=false, saveres=true);
    #catch
    #    println("a process has been interrupted")
    #end
end