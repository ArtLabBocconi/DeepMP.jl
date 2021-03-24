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
using Adapt
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
                                 reinfpar,
                                 verbose=1, 
                                 xtest=nothing,
                                 ytest=nothing)

    for it = 1:maxiters
        
        t = @timed Δ = update!(g, reinfpar)
        E = energy(g)

        verbose >= 1 && @printf("it=%d \t (r=%f) E=%d \t Δ=%f \n",
                                it, reinfpar.r, E, Δ)
        if verbose >= 2
            Etest = 100.0
            if ytest !== nothing
                Etest = mean(vec(forward(g, xtest)) .!= ytest) * 100
            end
            @printf("          Etest=%.2f%%  rstep=%g  t=%g\n", Etest, reinfpar.rstep, t.time)
        end

        plotinfo >=0 && plot_info(g, plotinfo, verbose=verbose)
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

function solve(; K::Vector{Int} = [101, 3],
                 Kteacher = K,
                 α = 100., # num_examples/num_params. Ignore if M is used.
                 M = -1, # num_params. If negative, set from alpha. 
                 Mtest = 10000, # num test samples
                 seedx::Int = -1,
                 density = 1,
                 TS = false, 
                 hidden_manifold = false, 
                 density_teacher = density,
                 kws...)

    seedx > 0 && Random.seed!(seedx)
    
    L = length(K) - 1
    density = process_density(density, L)
    numW = length(K)==2 ? K[1]*K[2]*density[1]  :
            sum(l->density[l] * K[l]*K[l+1], 1:length(K)-2)
    numW = round(Int, numW)

    if M <= 0 
        M = round(Int, α * numW)
        α = M / numW
    end
    
    N = K[1]
    D = Kteacher[1]
    @assert hidden_manifold || N == D 
    xtrain = rand(F[-1, 1], D, M)
    
    if TS
        teacher = rand_teacher(Kteacher; density=density_teacher)
        ytrain = Int.(forward(teacher, xtrain) |> vec)
        xtest = rand(F[-1, 1], D, M)
        ytest = Int.(forward(teacher, xtest) |> vec)
    else
        teacher = nothing
        ytrain = rand([-1,1], M)
        xtest, ytest = nothing, nothing
    end

    if hidden_manifold
        features = rand(F[-1, 1], N, D)    
        xtrain = sign.(features * xtrain ./ sqrt(D))
        if xtest !== nothing
            xtest = sign.(features * xtest ./ sqrt(D))
        end
    end

    @assert size(xtrain) == (N, M)
    @assert size(ytrain) == (M,)
    @assert all(x -> x == -1 || x == 1, ytrain)

    solve(xtrain, ytrain; K, density, teacher, xtest, ytest, kws...)
end


function solve(xtrain::AbstractMatrix, ytrain::AbstractVector;
                xtest = nothing, ytest = nothing,
                maxiters = 10000,
                ϵ = 1e-4,                      # convergence criterion
                K::Vector{Int} = [101, 3, 1],
                layers = [:tap, :tapex, :tapex],
                r = 0., rstep = 0.001,         # reinforcement parameters for W vars
                ψ = 0.,                        # damping coefficient
                yy = -1.,                      # focusing BP parameter
                h0 = nothing,                  # external field
                ρ = 1.,                        # coefficient for external field
                freezetop = true,              # freeze top-layer's weights to 1
                teacher = nothing,
                altsolv::Bool = true,
                altconv::Bool = false,
                seed::Int = -1, 
                plotinfo = 0,
                β = Inf,
                density = 1f0,                  # density of fully connected layer
                batchsize = -1,                 # only supported by some algorithms
                epochs = 100,
                verbose = 2,
                infotime = 10,
                usecuda = false,
                )

    usecuda = CUDA.functional() && usecuda
    device =  usecuda ? gpu : cpu
    if seed > 0
        Random.seed!(seed)
        usecuda && CUDA.seed!(seed)
    end
    
    xtrain, ytrain = device(xtrain), device(ytrain)
    xtest, ytest = device(xtest), device(ytest)
    dtrain = DataLoader((xtrain, ytrain); batchsize, shuffle=true, partial=false)

    g = FactorGraph(first(dtrain)..., K, layers; β, density, device)
    h0 !== nothing && set_external_fields!(g, h0; ρ);
    if teacher !== nothing
        teacher = device.(teacher)
        has_same_size(g, teacher) && set_weight_mask!(g, teacher)
    end
    initrand!(g)
    freezetop && freezetop!(g, 1)
    
    reinfpar = ReinfParams(r, rstep, yy, ψ)

    if batchsize <= 0
        
        it, e, δ = converge!(g; maxiters, ϵ, reinfpar,
                            altsolv, altconv, plotinfo,
                            teacher, verbose,
                            xtest, ytest)
        
    else
        
        ## MINI_BATCH message passing
        # TODO check reinfparams updates in mini-batch case
        
        #resfile = make_resfile(layers, K[1], K[2], batchsize, ρ, r, density)
        resfile = "results/res_Ks$(K)_bs$(batchsize)_layers$(layers)_rho$(ρ)_r$(r)_density$(density).dat"
        f = open(resfile, "w")

        for epoch = 1:epochs
            converged = solved = meaniters = 0
            t = @timed for (b, (x, y)) in enumerate(dtrain)
                ρ > 0 && set_Hext_from_H!(g, ρ)
                set_input_output!(g, x, y)

                it, e, δ = converge!(g; maxiters, ϵ, 
                                        reinfpar, altsolv, altconv, plotinfo,
                                        teacher, verbose=verbose-1)
                converged += (δ < ϵ)
                solved    += (e == 0)
                meaniters += it
                
                verbose >= 2 && print("b = $b / $(length(dtrain))\r")
            end

            Etrain = mean(vec(forward(g, xtrain)) .!= ytrain) * 100
            num_batches = length(dtrain)
            Etest = 100.0
            if ytest !== nothing
                Etest = mean(vec(forward(g, xtest)) .!= ytest) * 100
            end
            #
            mags_all = mags_symmetry(g, K)
            verbose > 1 && (println("mags overlaps="); display(mags_all); println())
            n_el = (K[2]^2-K[2])/2
            mag_ovrlp = ((sum(mags_all) - K[2])/2)/n_el
            #
            verbose >= 1 && @printf("Epoch %i (conv=%g, solv=%g <it>=%g): Etrain=%.2f%% Etest=%.2f%% mag_ov=%g r=%g rstep=%g ρ=%g  t=%g\n",
                                epoch, (converged/num_batches), (solved/num_batches), (meaniters/num_batches),
                                Etrain, Etest, mag_ovrlp, reinfpar.r, reinfpar.rstep, ρ, t.time)
            outf = @sprintf("%g %g %g", Etrain, Etest, mag_ovrlp)
            println(f, outf)
            flush(f)

            plot_info(g, 0, verbose=verbose)
            Etrain == 0 && break
        end
        close(f)
    end
    
    E = sum(vec(forward(g, xtrain)) .!= ytrain)
    return g, getW(g), teacher, E, it
    
end

end #module
