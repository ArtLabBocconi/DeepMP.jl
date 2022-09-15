multicl = [false]
datasets = [:fashion]
lays = [:tap]
seeds = [2]

K = [0, 501, 501, 0]
ρ = [1.0, 1.0, 0.9]
ψ = [0.99, 0.99, 0.99]
r = [0.0, 0.0, 0.0]
gpu_id = 2
#batchsize = 128

for multiclass in multicl, dataset in datasets, lay_type in lays, seed in seeds
    #try
        run_experiment(; multiclass, dataset, lay_type, seed, ρ,
        epochs=100, batchsize=Int(6e4), usecuda=true, gpu_id, 
        ψ, M=Int(6e4), maxiters=1, r, ϵinit=1.0, K, 
        altsolv=false, altconv=false, saveres=true);
    #catch
    #    println("a process has been interrupted")
    #end
end