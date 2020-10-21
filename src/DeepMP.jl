module DeepMP

using ExtractMacro
using FastGaussQuadrature
using SpecialFunctions
using Printf
using Random
using LinearAlgebra
using Statistics
using Base: @propagate_inbounds # for DataLoader
using LoopVectorization
using Tullio

# using PyPlot

const CVec = Vector{Complex{Float64}}
const IVec = Vector{Int}
const Vec = Vector{Float64}
const VecVec = Vector{Vec}
const IVecVec = Vector{IVec}
const VecVecVec = Vector{VecVec}
const IVecVecVec = Vector{IVecVec}

include("utils/utils.jl")
include("utils/functions.jl")
include("utils/dataloader.jl")
include("utils/Magnetizations.jl")
using .Magnetizations

include("layers/layers.jl")
include("factor_graph.jl")
include("reinforcement.jl")

function converge!(g::FactorGraph; maxiters::Int = 10000, ϵ::Float64=1e-5
                                , altsolv::Bool=false, altconv = false, plotinfo=0
                                , teacher = nothing
                                , reinfpar::ReinfParams=ReinfParams()
                                , verbose::Int=1)

    for it = 1:maxiters

        Δ = update!(g, reinfpar)
        E = energy(g)

        verbose > 0 && @printf("it=%d \t (r=%f ry=%f) E=%d \t Δ=%f \n",
                                it, reinfpar.r, reinfpar.ry, E, Δ)
        
        plotinfo >=0  && plot_info(g, plotinfo, verbose=verbose, teacher=teacher)
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

function solve(; K::Vector{Int} = [101,3], α=0.6,
                 seedx::Int=-1, 
                 density=1,
                 TS = false,
                 density_teacher = density,
                 kw...)

    seedx > 0 && Random.seed!(seedx)

    L = length(K) - 1
    density = process_density(density, L)
    numW = length(K)==2 ? K[1]*K[2]*density[1]  :
            sum(l->density[l] * K[l]*K[l+1], 1:length(K)-2)
    numW = round(Int, numW)

    N = K[1]
    M = round(Int, α * numW)
    xtrain = rand([-1.,1.], N, M)
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


function solve(xtrain::Matrix, ytrain::Vector{Int};
                xtest = nothing, ytest = nothing,
                maxiters::Int = 10000,
                ϵ::Float64 = 1e-4,              # convergence criteirum
                K::Vector{Int} = [101, 3, 1],
                layers=[:tap,:tapex,:tapex],
                r = 0., rstep = 0.001,          # reinforcement parameters for W vars
                ry = 0., rystep = 0.0,          # reinforcement parameters for Y vars
                ψ = 0.,                         # dumping coefficient
                yy = -1,                         # focusing BP parameter
                h0 = nothing,                   # external field
                ρ = 1.0,                        # coefficient for external field
                freezetop=true,                # freeze top-layer's weights to 1
                teacher = nothing,
                altsolv::Bool = true, 
                altconv::Bool = false,
                seed::Int = -1, plotinfo=0,
                β=Inf, βms = 1., 
                density = 1.,                   # density of fully connected layer
                batchsize=-1,                   # only supported by some algorithms
                epochs::Int = 1000,
                verbose::Int = 1)

    seed > 0 && Random.seed!(seed)
    
    g = FactorGraph(xtrain, ytrain, K, layers, β=β, βms=βms, density=density)
    h0 !== nothing && set_external_fields!(g, h0; ρ=ρ);
    teacher !== nothing && set_weight_mask!(g, teacher)
    initrand!(g)
    freezetop && freezetop!(g, 1)
    
    reinfpar = ReinfParams(r, rstep, ry, rystep, yy, ψ)

    if batchsize <= 0
        it, e, δ = converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar,
                            altsolv=altsolv, altconv=altconv, plotinfo=plotinfo,
                            teacher=teacher, verbose=verbose)
    else
        hext = get_allh(g)
        dtrain = DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true)
            
        for epoch = 1:epochs
            converged = solved = meaniters = 0
            for (b, (x, y)) in enumerate(dtrain)
                gbatch = FactorGraph(x, y, K, layers, β=β, βms=βms,
                                density=density, verbose=0)
                set_weight_mask!(gbatch, g)
                initrand!(gbatch)
                freezetop && freezetop!(g, 1)
                set_external_fields!(gbatch, hext; ρ=ρ)

                it, e, δ = converge!(gbatch, maxiters=maxiters, ϵ=ϵ, #reinfpar=ReinfParams(),
                                        reinfpar=reinfpar,
                                        altsolv=altsolv, altconv=altconv, plotinfo=plotinfo,
                                        teacher=teacher, verbose=verbose-1)
                converged += (δ < ϵ)
                solved    += (e == 0)
                meaniters += it
                copy_allh!(hext, gbatch)
                copy_mags!(g, gbatch)

                print("b = $b / $(length(dtrain))\r")
            end
            
            Etrain = mean(vec(forward(g, xtrain)) .!= ytrain) * 100
            num_batches = length(dtrain)
            Etest = 100.0
            if ytest != nothing
                Etest = mean(vec(forward(g, xtest)) .!= ytest) * 100
            end
            @printf("Epoch %i (conv=%g, solv=%g <it>=%g): Etrain=%.2f%% Etest=%.2f%% r=%g rstep=%g ρ=%g\n",
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
