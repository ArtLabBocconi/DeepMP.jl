multicl = [false]
datasets = [:fashion]
lays = [:bpi]
seeds = [2]

for multiclass in multicl, dataset in datasets, lay_type in lays, seed in seeds
    if lay_type ≠ :mf
        ρ = [1.0, 1.0, 1.0]
    else
        ρ = [1.0, 1.0, 1.0]
    end
    #try
        run_experiment(; multiclass, dataset, lay_type, seed, ρ,
        epochs=100, batchsize=Int(6e4), usecuda=true, gpu_id=1, 
        ψ=[0.0, 0.0, 0.0], M=Int(6e4), maxiters=1, r=0., ϵinit=1.0, K=[0, 101, 101, 0], 
        altsolv=false, altconv=false, saveres=true);
    #catch
    #    println("a process has been interrupted")
    #end
end