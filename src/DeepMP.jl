module DeepMP

using ExtractMacro
using FastGaussQuadrature
using SpecialFunctions
using Printf
using Random
using LinearAlgebra
using Statistics
using Base: @propagate_inbounds # for DataLoader
using Tullio
using LoopVectorization
using CUDA, KernelAbstractions
using Functors
CUDA.allowscalar(false)

# using PyPlot

const F = Float64
const CVec = Vector{Complex{F}}
const IVec = Vector{Int}
const Vec = Vector{F}
const VecVec = Vector{Vec}
const IVecVec = Vector{IVec}
const VecVecVec = Vector{VecVec}
const IVecVecVec = Vector{IVecVec}

include("cuda.jl")
include("utils/utils.jl")
include("utils/functions.jl")
include("utils/dataloader.jl")
include("utils/Magnetizations.jl"); using .Magnetizations
include("layers/layers.jl")
include("factor_graph.jl")
include("reinforcement.jl")

function converge!(g::FactorGraph;  maxiters=10000, ϵ=1f-5,
                                 altsolv=false, 
                                 altconv=false, 
                                 plotinfo=0,
                                 teacher=nothing,
                                 reinfpar::ReinfParams=ReinfParams(),
                                 verbose=1)

    for it = 1:maxiters
        Δ = update!(g, reinfpar)
        E = energy(g)

        verbose > 0 && @printf("it=%d \t (r=%f) E=%d \t Δ=%f \n",
                                it, reinfpar.r, E, Δ)

        plotinfo >=0 && plot_info(g, plotinfo, verbose=verbose, teacher=teacher)
        update_reinforcement!(reinfpar)
        if altsolv && E == 0
            verbose > 0 && println("Found Solution: correctly classified $(g.M) patterns.")
            return it, E, Δ
        end
        if altconv && Δ < ϵ
            verbose > 0 && println("Converged!")
            return it, E, Δ
        end
    end
    return maxiters, 1, 1.0
end

function solve(; K::Vector{Int}=[101,3], α=0.6,
                 seedx::Int=-1,
                 density=1,
                 TS=false,
                 density_teacher=density,
                 kw...)

    if seedx > 0
        Random.seed!(seedx)
        CUDA.seed!(seedx)
    end
    L = length(K) - 1
    density = process_density(density, L)
    numW = length(K)==2 ? K[1]*K[2]*density[1]  :
            sum(l->density[l] * K[l]*K[l+1], 1:length(K)-2)
    numW = round(Int, numW)

    N = K[1]
    M = round(Int, α * numW)
    xtrain = rand(F[-1, 1], N, M)
    if TS
        teacher = rand_teacher(K; density=density_teacher)
        ytrain = Int.(forward(teacher, xtrain) |> vec)
    else
        teacher = nothing
        ytrain = rand([-1,1], M)
    end

    @assert size(xtrain) == (N, M)
    @assert size(ytrain) == (M,)
    @assert all(x -> x == -1 || x == 1, ytrain)

    solve(xtrain, ytrain; K=K, density=density, teacher=teacher, kw...)
end


function solve(xtrain::AbstractMatrix, ytrain::AbstractVector;
                xtest = nothing, ytest = nothing,
                maxiters = 10000,
                ϵ = 1f-4,              # convergence criterium
                K::Vector{Int} = [101, 3, 1],
                layers=[:tap,:tapex,:tapex],
                r = 0f0, rstep = 0.001f0,          # reinforcement parameters for W vars
                ψ = 0f0,                         # dumping coefficient
                yy = -1,                         # focusing BP parameter
                h0 = nothing,                   # external field
                ρ = 1f0,                        # coefficient for external field
                freezetop=true,                # freeze top-layer's weights to 1
                teacher = nothing,
                altsolv::Bool = true,
                altconv::Bool = false,
                seed::Int = -1, plotinfo=0,
                β=Inf,
                density = 1f0,                   # density of fully connected layer
                batchsize=-1,                   # only supported by some algorithms
                epochs = 1000,
                verbose = 2,
                infotime=10,
                resfile="res.txt",
                usecuda = false,
                )

    if seed > 0
        Random.seed!(seed)
        CUDA.seed!(seed)
    end
    device = CUDA.has_cuda() && usecuda ? gpu : cpu
    
    xtrain, ytrain = device(xtrain), device(ytrain)
    xtest, ytest = device(xtest), device(ytest)
    dtrain = DataLoader((xtrain, ytrain); 
            batchsize, shuffle=true, partial=false)

    g = FactorGraph(first(dtrain)..., K, layers; β, density, device)
    h0 !== nothing && set_external_fields!(g, h0; ρ);
    if teacher !== nothing
        teacher = device.(teacher)
        set_weight_mask!(g, teacher)
    end
    initrand!(g)
    freezetop && freezetop!(g, 1)
    
    reinfpar = ReinfParams(r, rstep, yy, ψ)

    if batchsize <= 0
        it, e, δ = converge!(g; maxiters, ϵ, reinfpar,
                            altsolv, altconv, plotinfo,
                            teacher, verbose)
    else
        # TODO check reinfparams updates in mini-batch case
        
        for epoch = 1:epochs
            converged = solved = meaniters = 0
            for (b, (x, y)) in enumerate(dtrain)
                ρ > 0 && set_Hext_from_H!(g, ρ)
                set_input_output!(g, x, y)
                # ?? init!(gbatch) ??

                it, e, δ = converge!(g; maxiters, ϵ, 
                                        reinfpar, altsolv, altconv, plotinfo,
                                        teacher, verbose=verbose-1)
                converged += (δ < ϵ)
                solved    += (e == 0)
                meaniters += it
                
                verbose > 1 && print("b = $b / $(length(dtrain))\r")
            end

            Etrain = mean(vec(forward(g, xtrain)) .!= ytrain) * 100
            num_batches = length(dtrain)
            Etest = 100.0
            if ytest != nothing
                Etest = mean(vec(forward(g, xtest)) .!= ytest) * 100
            end
            verbose > 0 &&  @printf("Epoch %i (conv=%g, solv=%g <it>=%g): Etrain=%.2f%% Etest=%.2f%% r=%g rstep=%g ρ=%g\n",
                                epoch, (converged/num_batches), (solved/num_batches), (meaniters/num_batches),
                                Etrain, Etest, reinfpar.r, reinfpar.rstep, ρ)
            plot_info(g, 0, verbose=verbose, teacher=teacher)
            Etrain == 0 && break
        end
    end

    E = sum(vec(forward(g, xtrain)) .!= ytrain)
    return g, getW(g), teacher, E, it
end

end #module
