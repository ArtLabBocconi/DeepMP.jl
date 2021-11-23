multicl = [true]
gpu_id = 2
epochs = 100
#datasets = [:fashion]
#lays = [:bpi, :tap, :mf]
#seeds = [2, 7, 11]
#Ks = [[0, 101, 101, 0], [0, 501, 501, 501, 0]]
#ρs = [[1.0, 1.0, 0.9], [1.0, 1.0, 1.0, 0.9]]
#ψs = [[0.8, 0.8, 0.8], [0.8, 0.8, 0.8, 0.8]]

datasets = [:fashion]
lays = [:mf]
seeds = [2, 7, 11]
#Ks = [[0, 1024, 1024, 1024, 1024, 1024, 0]]
Ks = [[0, 501, 501, 501, 501, 0]]
ρs = [[1.0+1e-4, 1.0+1e-4, 1.0+1e-4, 1.0+1e-4, 0.9]]
#ψs = [[0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]
ψs = [[0.2, 0.2, 0.2, 0.2, 0.999]]

density = 1.0

for multiclass in multicl, dataset in datasets, (K, ρ, ψ) in zip(Ks, ρs,  ψs), lay_type in lays, seed in seeds
    #try
        run_experiment(; multiclass, dataset, lay_type, seed, ρ, density,
        epochs, batchsize=128, usecuda=true, gpu_id, ψ, 
        M=Int(6e4), maxiters=1, r=0.0, ϵinit=1.0, K, 
        altsolv=false, altconv=false, saveres=true);
    #catch
    #    println("a process has been interrupted")
    #end
end