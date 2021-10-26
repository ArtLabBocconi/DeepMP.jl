multicl = [true]
datasets = [:fashion]
lays = [:mf]
seeds = [2]

for multiclass in multicl, dataset in datasets, lay_type in lays, seed in seeds
    ρ = [1.0001, 1.0001, 0.9]
    #try
        run_experiment(; multiclass, dataset, lay_type, seed, ρ,
        epochs=100, batchsize=Int(6e4), usecuda=true, gpu_id=1, 
        ψ=[0.81, 0.81, 0.81], M=Int(6e4), maxiters=1, r=0.0, ϵinit=1.0, K=[0, 501, 501, 0], 
        altsolv=false, altconv=false, saveres=true);
    #catch
    #    println("a process has been interrupted")
    #end
end