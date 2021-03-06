multicl = [false]
datasets = [:mnist]
lays = [:bpi]
seeds = [2]
gpu_id = 1

usecuda = true

for multiclass in multicl, dataset in datasets, lay_type in lays, seed in seeds
    if lay_type ≠ :mf
        #ρ = [0.9999, 0.9999, 0.9]
        ρ = [1.0001, 1.0001, 0.9]
    else
        ρ = [1.0+1e-4, 1.0+1e-4, 0.]
    end
    #try
        run_experiment(; multiclass, dataset, lay_type, seed, ρ,
        epochs=100, batchsize=128, usecuda, gpu_id, 
        ψ=[0.8, 0.8, 0.8], M=Int(6e4), maxiters=1, r=0.0, ϵinit=1.0, K=[0, 101, 101, 0], 
        altsolv=false, altconv=false, saveres=true);
    #catch
    #    println("a process has been interrupted")
    #end
end