multicl = [true]
datasets = [:fashion]
lays = [:bpi]
seeds = [2, 7, 11, 19, 27]

for multiclass in multicl, dataset in datasets, lay_type in lays, seed in seeds
    if lay_type ≠ :mf
        ρ = [1.0+1e-4, 1.0+1e-4, 1.0+1e-4]
        ρ = [1.0, 1.0, 1.0-1e-3]
        ρ = [1.0, 1.0, 1.0] .+ 1e-3
    else
        ρ = [1.0, 1.0, 1.0]
        #ρ = [1.0-1e-3, 1.0-1e-3, 1.0-1e-3]
    end
    #try
        run_experiment(; multiclass, dataset, lay_type, seed, ρ,
        epochs=1, batchsize=1, usecuda=true, gpu_id=2, 
        ψ=[0., 0., 0.], M=Int(5e4), maxiters=1, r=0., ϵinit=1.0, K=[0, 101, 101, 0], 
        altsolv=false, altconv=false, saveres=true);
    #catch
    #    println("a process has been interrupted")
    #end
end