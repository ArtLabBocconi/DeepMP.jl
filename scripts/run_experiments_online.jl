multicl = [true]
datasets = [:fashion]
lays = [:mf]
seeds = [2]

for multiclass in multicl, dataset in datasets, lay_type in lays, seed in seeds
    if lay_type ≠ :mf
        ρ = [1.0+1e-4, 1.0+1e-4, 1.0+1e-4]
        ρ = [1.0, 1.0, 1.0]
    else
        ρ = [1.0, 1.0, 1.0]
        #ρ = [1.0-1e-3, 1.0-1e-3, 1.0-1e-3]
    end
    #try
        run_experiment(; multiclass, dataset, lay_type, seed, ρ,
        epochs=3, batchsize=1, usecuda=true, gpu_id=0, 
        ψ=[0.0, 0.0, 0.0], M=Int(1e3), maxiters=1, r=0., ϵinit=1.0, K=[0, 101, 101, 0], 
        altsolv=false, altconv=false, saveres=true);
    #catch
    #    println("a process has been interrupted")
    #end
end