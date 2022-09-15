multicl = [false]
datasets = [:fashion]
lays = [:cbpi]
seeds = [2]
gpu_id = 2
density = 0.05

K = [0, 501, 501, 0]
ρ = [1.0, 1.0, 1.0]
ψ = [0.8, 0.8, 0.8]
density = [0.1, 0.1, 0.1]

batchsize = 128
if batchsize > 0
    epochs = 100
    maxiters = 1
    r = 0.0
else
    epochs = 1
    maxiters = 100
    r = 0.99
end

for multiclass in multicl, dataset in datasets, lay_type in lays, seed in seeds
    #try
        run_experiment(; multiclass, dataset, lay_type, seed, ρ, density,
        epochs, batchsize, usecuda=true, gpu_id, ψ, 
        M=Int(6e4), maxiters, r, ϵinit=1.0, K, 
        altsolv=false, altconv=false, saveres=true);
    #catch
    #    println("a process has been interrupted")
    #end
end