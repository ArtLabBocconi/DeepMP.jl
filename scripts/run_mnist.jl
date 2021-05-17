using Distributed
distributed = true

usecuda = true
if length(ARGS) ≠ 0
    gpu_id = parse(Int, ARGS[1])
else
    gpu_id = 0
end
epochs = 100
lays = [:bp, :bpi, :tap]

batchsize = 128
M = Int(6e4)
#ρs = [-1e-1, -1e-5, 0., 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] .+ 1.

ρs = [1e-4]
ψs = [0:0.2:1;]
@show ψs 
error()

maxiters = 1
r = 0.

K = [28*28, 101, 101, 1]

# TODO: pmap non funziona bene, forse per la gpu?
# TODO: usare un dizionario
params = []
for lay in lays, ρ in ρs, ψ in ψs
    push!(params, [lay, ρ, ψ])
end
total_procs = length(params)

if distributed

    max_procs = 30
    if total_procs < max_procs
        addprocs(total_procs+1)
    else
        addprocs(max_procs+1)
    end

    @everywhere include("mnist.jl")
    pmap(p -> run_experiment(9; usecuda, gpu_id, epochs, lay=p[1], 
                                batchsize, ρ=p[2], ψ=p[3], M, maxiters, r, K), params; on_error=x->0)
else
    include("mnist.jl")
    for p in params
        try
            run_experiment(9; usecuda, gpu_id, epochs, lay=p[1], 
                              batchsize, ρ=p[2], ψ=p[3], M, maxiters, r, K)
        catch
            println("a process ha been interrupted")
        end
    end
end