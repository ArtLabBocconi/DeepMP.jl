#multicl = [false]
#datasets = [:fashion]
#lays = [:bp] 
#gpu_id = 0
#seeds = [2]

#K = [0, 501, 501, 0]
#ρ = [1.0, 1.0, 0.9]
#ψ = [0.8, 0.8, 0.8]

#density = 1.0

#for multiclass in multicl, dataset in datasets, lay_type in lays, seed in seeds
    #try
        run_experiment(; multiclass=false, dataset=:fashion, lay_type=:bp, seed=2, 
            ρ=[1.0, 1.0, 0.9], density=1.0, ψ=[0.8, 0.8, 0.8],
            epochs=200, batchsize=128, usecuda=true, gpu_id=0,  
            M=Int(6e4), maxiters=1, r=0.0, ϵinit=1.0, K=[28*28, 501, 501, 1], 
            altsolv=false, altconv=false, saveres=true);
    #catch
    #    println("a process has been interrupted")
    #end
#end
