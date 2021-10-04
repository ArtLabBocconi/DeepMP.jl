multicl = [false]
datasets = [:fashion]
lays = [:tap]
seeds = [2]

for multiclass in multicl, dataset in datasets, lay_type in lays, seed in seeds
    if lay_type ≠ :mf
        ρ = [1.0, 1.0, 1.0]
    else
        ρ = [1.0, 1.0, 1.0]
    end
    #try
        run_experiment(; multiclass, dataset, lay_type, seed, ρ,
        epochs=1, batchsize=-1, usecuda=true, gpu_id=2, 
        ψ=[0.99, 0.99, 0.99], M=Int(6e4), maxiters=100, r=[0.9, 0.9, 0.9],
        ϵinit=1.0, K=[0, 101, 101, 0], 
        altsolv=false, altconv=false, saveres=true, verbose=2);
    #catch
    #    println("a process has been interrupted")
    #end
end