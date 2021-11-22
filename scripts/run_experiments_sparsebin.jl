multicl = [true]
datasets = [:mnist]
lays = [:tap for _ = 1:100]
seeds = [2]
gpu_id = 1
densities = [0.1:0.1:1.0...]

K = [0, 101, 101, 0]
#ρs = [[1.0+1e-4, 1.0+1e-4, 0.9], [1.0+1e-4, 1.0+1e-4, 0.9], [1.0+1e-4, 1.0+1e-4, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.9], [1.0+1e-4, 1.0+1e-4, 0.9], [1.0, 1.0, 0.0], [1.0+1e-4, 1.0+1e-4, 0.0]]
#ψs = [[0.8, 0.8, 0.99999], [0.8, 0.8, 0.8], [0.8, 0.8, 0.99], [0.8, 0.8, 0.99], [0.8, 0.8, 0.99], [0.8, 0.8, 0.99], [0.8, 0.8, 0.9999], [0.8, 0.8, 0.9999]]

ρs = [[1.0, 1.0, 0.9], [1.0, 1.0, 0.9], [1.0, 1.0, 0.9], [1.0, 1.0, 0.0], [1.0+1e-4, 1.0+1e-4, 0.0], [1.0, 1.0, 0.0], [1.0+1e-4, 1.0+1e-4, 0.0]]
ψs = [[0.2, 0.2, 0.99999], [0.2, 0.2, 0.2], [0.2, 0.2, 0.9], [0.8, 0.8, 0.99999], [0.8, 0.8, 0.99999], [0.8, 0.8, 0.999999], [0.8, 0.8, 0.999999]]

for multiclass in multicl, dataset in datasets, (lay_type, ρ, ψ) in zip(lays, ρs, ψs), density in densities, seed in seeds
    #try
        run_experiment(; multiclass, dataset, lay_type, seed, ρ, density,
        epochs=100, batchsize=128, usecuda=true, gpu_id, ψ, 
        M=Int(6e4), maxiters=1, r=0.0, ϵinit=1.0, K, 
        altsolv=false, altconv=false, saveres=true);
    #catch
    #    println("a process has been interrupted")
    #end
end