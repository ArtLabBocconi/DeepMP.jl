multicl = [false, true]
datasets = [:mnist, :fashion, :cifar10]
#lays = [:bpi]: gpu_id = 0
#lays = [:tap, :mf]; gpu_id = 0
lays = [:bpi, :tap, :mf]; gpu_id = 0
seeds = [2, 7, 11]

K = [0, 1024, 1024, 1024, 0]
ρ = [1.0, 1.0, 1.0, 0.9]
ψ = [0.8, 0.8, 0.8, 0.8]

density = 1.0

for multiclass in multicl, dataset in datasets, lay_type in lays, seed in seeds
    #try
        run_experiment(; multiclass, dataset, lay_type, seed, ρ, density,
        epochs=200, batchsize=128, usecuda=true, gpu_id, ψ, 
        M=Int(6e4), maxiters=1, r=0.0, ϵinit=1.0, K, 
        altsolv=false, altconv=false, saveres=true);
    #catch
    #    println("a process has been interrupted")
    #end
end