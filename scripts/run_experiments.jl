using Distributed
distributed = false

usecuda = true
if length(ARGS) ≠ 0
    gpu_id = parse(Int, ARGS[1])
end
#lays = [:mf, :bp]; gpu_id = 1
#lays = [:bpi, :tap]; gpu_id = 2
#lays = [:mf, :tap]; gpu_id = 1
#lays = [:bpi]; gpu_id = 2

lays = [:bpi]; gpu_id = 2
#lays = [:bpi]; gpu_id = 2


epochs = 100

dataset = :fashion

altsolv = true
altconv = true

# parametri selezionati
Ks = [[28*28, 101, 101, 1]]
ρs = [1e-5] .+ 1.
ϵinits = [1e0]
ψs = [0.8]
maxiterss = [2]
rs = [0.]

Ms = [Int(6e4)]
bs = [128]

#ρs = [-1e-1, -1e-5, 0., 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] .+ 1.
#ψs = [[0:0.2:0.8;]..., 0.9, 0.99, 0.999, 0.9999]
#bs = [1, 16, 128, 1024]; Ms = [Int(6e4) for _=1:length(bs)]
#Ms = [Int(1e2), Int(1e3), Int(1e4), Int(6e4)]; bs = [Int(1e0), Int(1e1), Int(1e2), Int(6e2)]
#maxiterss = [1, 10, 50, 100]
#rs = [0.2:0.2:1.2;]
#ϵinits = [0., 0.01, 0.1, 0.5, 1., 1.5, 2., 2.5, 3.]
#Ks = [[28*28, 101, 1], [28*28, 501, 1], [28*28, 1001, 1], 
#      [28*28, 101, 101, 1], [28*28, 501, 501, 1], [28*28, 1001, 1001, 1], 
#      [28*28, 101, 101, 101, 1], [28*28, 501, 501, 501, 1], [28*28, 1001, 1001, 1001, 1]]

# TODO: pmap non funziona bene, forse per la gpu? la gpu potrebbe riempirsi troppo. bisogna occuparsene.

# TODO: usare un dizionario
params = []
for lay in lays, ρ in ρs, ψ in ψs, (M, b) in zip(Ms, bs), maxiters in maxiterss, r in rs, ϵinit in ϵinits, K in Ks
    if lay == :mf
        ρ1 = 10*ρ 
    else
        ρ1 = ρ
    end
    push!(params, [lay, ρ1, ψ, M, b, maxiters, r, ϵinit, K])
end
total_procs = length(params)

if distributed

    max_procs = 30
    # TODO: non voglio continuare ad aggiungere processi se ci sono già i processi necessari
            # fallo con npropcs()
    procs_to_open = min(total_procs, max_procs)
    addprocs(procs_to_open)
            
    @everywhere include("real_data_experiments.jl")
    pmap(p -> run_experiment(9; usecuda, gpu_id, epochs, layers=[p[1] for _=1:length(p[9])-1], 
              batchsize=p[5], ρ=p[2], ψ=p[3], M=p[4], maxiters=p[6], r=p[7], ϵinit=p[8], 
              K=p[9], altsolv, altconv, dataset), 
              params; on_error=x->0)
else
    include("real_data_experiments.jl")
    for p in params
        try
            run_experiment(; dataset, multiclass=false, usecuda, gpu_id, epochs, layers=[p[1] for _=1:length(p[9])-1], 
            batchsize=p[5], ρ=p[2], ψ=p[3], M=p[4], maxiters=p[6], r=p[7], 
            ϵinit=p[8], K=p[9], altsolv, altconv)
        catch
            println("a process has been interrupted")
        end
    end
end