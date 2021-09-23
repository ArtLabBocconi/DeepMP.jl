using JLD2

multicl = [false]
datasets = [:fashion]
lays = [:bpi]
seeds = [2]

for multiclass in multicl, dataset in datasets, lay_type in lays, seed in seeds
    if lay_type ≠ :mf
        ρ = [1.0, 1.0, 0.9]
    else
        ρ = [1.0+1e-4, 1.0+1e-4, 0.]
    end
    #try
        g, ws, teacher, Etrain, it, conf_file = 
        run_experiment(; multiclass, dataset, lay_type, seed, ρ,
        epochs=10, batchsize=128, usecuda=true, gpu_id=1, 
        ψ=[0.8, 0.8, 0.8], M=Int(6e4), maxiters=1, r=0.0, ϵinit=1.0, K=[0, 101, 101, 0], 
        altsolv=false, altconv=false, saveres=true);
        
        ws = map(Array, ws)

        save(conf_file, Dict("weights" => ws))

    #catch
    #    println("a process has been interrupted")
    #end
end