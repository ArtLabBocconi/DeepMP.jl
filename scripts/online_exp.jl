# ONLINE Training: Message Passing vs Gradient Descent

module online_exp

using Statistics, Random, LinearAlgebra, DelimitedFiles

include("real_data_experiments.jl")

using CUDA, KernelAbstractions, CUDAKernels
using Functors
using Adapt

cpu(x) = fmap(x -> adapt(Array, x), x)

gpu(x) = fmap(CUDA.cu, x)
# CUDA.cu(x::Integer) = x
CUDA.cu(x::Float64) = Float32(x)
CUDA.cu(x::Array{Int64}) = convert(CuArray{Int32}, x)

include("../../DeepBinaryNets/src/DeepBinaryNets.jl")

signB(x::T) where {T} = sign(x + 1f-10)

# Binary Classification
function forward(w, x)
    L = length(w)
    for l = 1:L
        x = signB.(w[l] * x)
    end
    return vec(x)
end
# Multiclass
function forward_mc(w, x)
    L = length(w)
    for l = 1:(L-1)
        x = signB.(w[l] * x)
    end
    ŷ = argmax(w[L] * x, dims=1)
    return vec(getindex.(ŷ, 1))
end

function errors(w, x, y)
    if maximum(y) > 1
        return mean(forward_mc(w, x) .!= y)
    else
        return mean(forward(w, x) .!= y)
    end
end

function run(Prange, seeds;
                    seed_data=17,
                    H::Vector{Int}=[21],
                    dataset=:mnist, classes=[], multiclass=true,
                    lr=0.1, density=1.0,
                    layers=[:bpi, :bpi], maxiters=5,
                    ρ=1.0, r=0.0, rstep=0.0, ψ=0.0, ϵinit=0.01,
                    altsolv=true, altconv=true,
                    usecuda=false, gpu_id=-1,
                    outfile="tmp.dat");

    usecuda = CUDA.functional() && usecuda
    device = usecuda ? gpu : cpu
    usecuda && gpu_id >= 0 && device!(gpu_id)

    x, y, xt, yt = get_dataset(-1; seed=seed_data,
                                  multiclass=multiclass, dataset=dataset)
    x, y, xt, yt = device(x), device(y), device(xt), device(yt)

    f = open(outfile, "w")
    for P in Prange
        trnerrs_bp  = []
        tsterrs_bp  = []
        trnerrs_sgd = []
        tsterrs_sgd = []
        for seed in seeds
            idxs = randperm(size(x, 2))[1:P]
            xp = x[:,idxs]
            yp = y[idxs]

            # SGD
            lossv = multiclass ? "xent" : "binxent"
            ws, _ = DeepBinaryNets.main(seed=21,
                                xtrn=xp, ytrn=yp, xtst=xt, ytst=yt,
                                model="mlp", h=[H...], w0=nothing,
                                epochs=1, B=1, lr=lr,
                                classes=classes, lossv=lossv, earlystop=false,
                                losstol=10.0, nrep=1,
                                pdrop=0.0, opt="SGD", comp_loss=false);
            # BP
            outnode = multiclass ? 1 : 10
            @show size(xp), size(yp), outnode
            g, wb, wt, E, it = DeepMP.solve(xp, yp;
                         K=[size(xp, 1), H..., 10], layers=layers,
                         xtest=xt, ytest=yt, ϵinit=ϵinit,
                         ρ=ρ, r=r, rstep=rstep, yy=0.0,
                         seed=23, epochs=1, maxiters=maxiters,
                         ψ=ψ, batchsize=1,
                         ϵ=1e-4, altsolv=altsolv, altconv=altconv,
                         freezetop=false,
                         usecuda=usecuda, gpu_id=gpu_id,
                         verbose=1);

            push!(trnerrs_bp,  errors(wb, xp, yp))
            push!(tsterrs_bp,  errors(wb, xt, yt))
            push!(trnerrs_sgd, errors(ws, xp, yp))
            push!(tsterrs_sgd, errors(ws, xt, yt))
        end # seeds
        norm = 1.0 / sqrt(length(seeds) - 1.0)
        E_bp = mean(trnerrs_bp)
        ΔE_bp = std(trnerrs_bp) * norm
        Et_bp = mean(tsterrs_bp)
        ΔEt_bp = std(tsterrs_bp) * norm
        E_sgd = mean(trnerrs_sgd)
        ΔE_sgd = std(trnerrs_sgd) * norm
        Et_sgd = mean(tsterrs_sgd)
        ΔEt_sgd = std(tsterrs_sgd) * norm
        # println("$P $(E_bp) $(ΔE_bp) $(Et_bp) $(ΔEt_bp) $(E_sgd) $(ΔE_sgd) $(Et_sgd) $(ΔEt_sgd)")
        println(f, "$P $(E_bp) $(ΔE_bp) $(Et_bp) $(ΔEt_bp) $(E_sgd) $(ΔE_sgd) $(Et_sgd) $(ΔEt_sgd)")
        flush(f)
    end # Prange
    close(f)

end # run

end # module
