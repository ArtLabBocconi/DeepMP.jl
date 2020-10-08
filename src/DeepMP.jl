module DeepMP

using ExtractMacro
using FastGaussQuadrature
using SpecialFunctions
using Printf
using Random
using LinearAlgebra
using Statistics
# using Flux

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
include("utils/Magnetizations.jl")
using .Magnetizations

include("layers.jl")
include("dropout.jl")
include("factor_graph.jl")
include("reinforcement.jl")

function converge!(g::FactorGraph; maxiters::Int = 10000, ϵ::Float64=1e-5
                                , altsolv::Bool=false, altconv = false, plotinfo=0
                                , teacher = nothing
                                , reinfpar::ReinfParams=ReinfParams()
                                , verbose::Int=1)

    for it = 1:maxiters

        Δ = update!(g, reinfpar)
        E, h = energy(g)

        # verbose > 0 && @printf("it=%d \t params E=%d \t Δ=%f \n",
        #                         it, reinfpar.r, reinfpar.ry, E, Δ)
        verbose > 0 && (reinfpar.y > 0 ?
                        @printf("it=%d \t (pol=%.3f, y=%.1f) \t E=%d \t Δ=%f \n",
                                 it, tanh(reinfpar.r), reinfpar.y, E, Δ) :
                        @printf("it=%d \t (r=%.3f, ry=%.3f) \t E=%d \t Δ=%f \n",
                                 it, reinfpar.r, reinfpar.ry, E, Δ))
        # println(h)
        plotinfo >=0  && plot_info(g, plotinfo, verbose=verbose, teacher=teacher)
        update_reinforcement!(reinfpar)
        if altsolv && E == 0
            verbose > 0 && println("Found Solution: correctly classified $(g.M) patterns.")
            return E, Δ
            #break
        end
        if altconv && Δ < ϵ
            verbose > 0 && println("Converged!")
            return E, Δ
            #break
        end
    end
    return 1, 1.0
end

function rand_teacher(K::Vector{Int}; density=1.)
    L = length(K)-1
    @assert K[L+1] == 1

    if isa(density, Number)
        density = fill(density, L)
    end
    @assert length(density) == L

    T = Float64
    W = Vector{Vector{Vector{T}}}()
    for l=1:L
        push!(W, [rand(T[-1,1], K[l]) for k=1:K[l+1]])
        for k in 1:K[l+1]
            W[l][k] .*= [rand() < density[l] ? 1 : 0 for i=1:K[l]]
        end
    end
    if L > 1
        W[L][1] .= 1
    end
    return W
end

function solveTS(; K::Vector{Int} = [101,3], α::Float64=0.6,
                   seedξ::Int=-1,
                   density = 1,
                   density_teacher = density,
                   kw...)
    seedξ > 0 && Random.seed!(seedξ)
    numW = length(K)==2 ? K[1]*K[2]  : sum(l->K[l]*K[l+1],1:length(K)-2)
    N = K[1]
    ξ = zeros(K[1], 1)
    M = round(Int, α * numW)
    ξ = rand([-1.,1.], K[1], M)
    W = rand_teacher(K; density=density_teacher)
    σ = Int[forward(W, ξ[:, a])[1][1] for a=1:M]
    @assert (any(i -> i == 0, σ) == false)

    solve(ξ, σ; K=K, teacher=W, density=density, kw...)
end

function solve(; K::Vector{Int} = [101,3], α=0.6,
                 seedξ::Int=-1, realξ = false,
                 dξ::Vector{Float64} = Float64[], nξ::Vector{Int} = Int[],
                 maketree = false, kw...)

    seedξ > 0 && Random.seed!(seedξ)
    numW = length(K)==2 ? K[1]*K[2]  : sum(l->K[l]*K[l+1],1:length(K)-2)
    maketree && (numW = div(numW, K[2]))
    N = K[1]
    ξ = zeros(K[1], 1)

    M = round(Int, α * numW)
    if realξ
        ξ = randn(K[1], M)
    else
        ξ = rand([-1.,1.], K[1], M)
    end
    # σ = ones(Int, M)
    σ = rand([-1,1], M)
    @assert size(ξ) == (N, M)
    # println("Mean Overlap ξ $(meanoverlap(ξ))")
    solve(ξ, σ; K=K, maketree=maketree, kw...)
end

function solveMNIST(; α=0.01, K::Vector{Int} = [784,10], kw...)
    @assert K[1] == 28*28
    # @assert K[end] == 10
    N = 784; M=round(Int, α*60000)
    h5 = h5open("data/mnist/train.hdf5", "r")
    ξ0 = reshape(h5["data"][:,:,1,1:M], N, M)
    m = mean(ξ0)
    m1, m2 = minimum(ξ0), maximum(ξ0)
    Δ = max(abs(m1-m), abs(m2-m))
    ξ = zeros(N, M)
    for i=1:N, a=1:M
        ξ[i,a] = (ξ0[i,a] - m) / Δ
    end
    @assert all(-1 .<= ξ .<= 1.)
    σ = round(Int, reshape(h5["label"][:,1:M], M) + 1)
    σ = Int[σ==1 ? 1 : -1 for σ in σ]
    solve(ξ, σ; K=K, kw...)
end

# function solve(ξ::Matrix, σ::Vector{Int};
#                 maxiters::Int = 10000, ϵ::Float64 = 1e-4,
#                 epochs::Int = 10000,
#                 K::Vector{Int} = [101, 3, 1],layers=[:tap,:tapex,:tapex],
#                 r = 0., rstep = 0.001,
#                 ry = 0., rystep = 0.0,
#                 ψ = 0., # dumping coefficient
#                 y = -1, # focusing
#                 teacher::Union{VecVecVec, Nothing} = nothing,
#                 altsolv::Bool = true, altconv::Bool = false,
#                 seed::Int = -1, plotinfo=0,
#                 β=Inf, βms = 1., rms = 1., ndrops = 0, maketree=false,
#                 density = 1., # density of fully connected layer
#                 use_teacher_weight_mask = true,
#                 batchsize=-1, # only supported by some algorithms
#                 verbose::Int = 1)
#
#     seed > 0 && Random.seed!(seed)
#     g = FactorGraph(ξ, σ, K, layers, β=β, βms=βms, rms=rms, ndrops=ndrops, density=density)
#     if use_teacher_weight_mask
#         set_weight_mask!(g, teacher)
#     end
#     initrand!(g)
#     fixtopbottom!(g)
#     maketree && maketree!(g.layers[2])
#     reinfpar = ReinfParams(r, rstep, ry, rystep, y, ψ)
#
#     if batchsize <= 0
#         converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar,
#                 altsolv=altsolv, altconv=altconv, plotinfo=plotinfo,
#                 teacher=teacher, verbose=verbose)
#     else
#         dtrain = Flux.Data.DataLoader((ξ, σ), batchsize=batchsize, shuffle=true)
#         for epoch=1:epochs
#             for (x, y) in dtrain
#                 gbatch = FactorGraph(x, y, K, layers, β=β, βms=βms,
#                                 rms=rms, ndrops=ndrops, density=density, verbose=0)
#                 set_weight_mask!(gbatch, g)
#                 initrand!(gbatch)
#                 fixtopbottom!(gbatch)
#
#                 for l=2:gbatch.L+1
#                     for k in 1:g.layers[l].K
#                         gbatch.layers[l].allhext[k] .= g.layers[l].allhext[k]
#                         # gbatch.layers[l].allh[k] .= g.layers[l].allh[k]
#                     end
#                 end
#
#                 converge!(gbatch, maxiters=maxiters, ϵ=ϵ, reinfpar=ReinfParams(),
#                     altsolv=false, altconv=true, plotinfo=plotinfo,
#                     teacher=teacher, verbose=0)
#
#                 for l=2:gbatch.L+1
#                     for k in 1:g.layers[l].K
#                         @assert all(isfinite, gbatch.layers[l].allh[k])
#                         # g.layers[l].allhext[k] .= reinfpar.r * gbatch.layers[l].allh[k]
#                         g.layers[l].allhext[k] .= gbatch.layers[l].allh[k]
#                         # g.layers[l].allh[k] .= gbatch.layers[l].allh[k]
#                         g.layers[l].allm[k] .= tanh.(gbatch.layers[l].allh[k])
#                     end
#                 end
#                 fixtopbottom!(g)
#             end
#             E, stab = energy(g)
#
#             println("Epoch $epoch: E=$E r=$(reinfpar.r)  rstep=$(reinfpar.rstep)")
#             update_reinforcement!(reinfpar)
#             plot_info(g, 0, verbose=verbose, teacher=teacher)
#             altsolv && (E==0) && break
#         end
#     end
#
#     E, stab = energy(g)
#     return g, getW(g), teacher, E, stab
# end

function solve(ξ::Matrix, σ::Vector{Int}; maxiters::Int = 10000, ϵ::Float64 = 1e-4,
                K::Vector{Int} = [101, 3, 1],layers=[:tap,:tapex,:tapex],
                r = 0., rstep = 0.001,
                ry = 0., rystep = 0.0,
                ψ = 0., # dumping coefficient
                y = -1, # focusing
                teacher::Union{VecVecVec, Nothing} = nothing,
                altsolv::Bool = true, altconv::Bool = false,
                seed::Int = -1, plotinfo=0,
                β=Inf, βms = 1., rms = 1., ndrops = 0, maketree=false,
                density = 1., # density of fully connected layer
                use_teacher_weight_mask = false,
                batchsize=-1, # only supported by some algorithms
                epochs::Int = 1000,
                verbose::Int = 1,
                verbose_in::Int = 0)

    seed > 0 && Random.seed!(seed)

    if batchsize <= 0
        g = FactorGraph(ξ, σ, K, layers, β=β, βms=βms, rms=rms, ndrops=ndrops, density=density)
        if use_teacher_weight_mask
            set_weight_mask!(g, teacher)
        end
        initrand!(g)
        fixtopbottom!(g)
        maketree && maketree!(g.layers[2])
        reinfpar = ReinfParams(r, rstep, ry, rystep, y, ψ)

        converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar,
                  altsolv=altsolv, altconv=altconv, plotinfo=plotinfo,
                  teacher=teacher, verbose=verbose)
    else
        hext = [[0.0.*rand(K[l]) for k = 1:K[l+1]] for l = 1:length(K)-1]

        N, M = size(ξ)
        @info "Num pattern = $M (N=$N)"
        last = (batchsize > M ? M : batchsize)
        batchsize > M && (@warn "Batchsize ($batchsize) > M ($M)")
        initbatch = randperm(M)[1:last]
        g = FactorGraph(ξ[:,initbatch], σ[initbatch], K, layers, β=β, βms=βms,
                        rms=rms, ndrops=ndrops, density=density, verbose=0)
        if use_teacher_weight_mask
            set_weight_mask!(g, teacher)
        end
        initrand!(g)
        fixtopbottom!(g)
        reinfpar = ReinfParams(r, rstep, ry, rystep, y, ψ)

        num_batches = M ÷ batchsize + ((M % batchsize) == 0 ? 0 : 1)
        for epoch = 1:epochs
            converged = 0
            solved = 0
            patt_perm = randperm(M)
            for b = 1:num_batches
                first = (batchsize*(b-1)+1)
                last = (batchsize * b > M ? M : batchsize * b)
                batch = patt_perm[first:last]

                # g.ξ = ξ[:,batch]
                # g.σ = σ[batch]
                # g.M = length(batch)
                # g.layers[1] = InputLayer(g.ξ)
                # g.layers[end] = OutputLayer(g.σ, β=β)
                # for l=2:g.L+1
                #     for k in 1:g.layers[l].K
                #         g.layers[l].allh[k] .= 0.0
                #         #g.layers[l].allh[k] .= 0.1 * rand(K[l-1])
                #     end
                #     for m in 1:g.M
                #         g.layers[l].allhy[m] .= 0.0
                #     end
                # end
                # for l = 1:g.L+1
                #     chain!(g.layers[l], g.layers[l+1])
                # end
                # initrand!(g)
                # fixtopbottom!(g)

                # SAME AS ABOVE, less clear and redundant ops,
                # but in the explicit way I have issues with varying batchsize..
                w = getW(g)
                g = FactorGraph(ξ[:,batch], σ[batch], K, layers, β=β, βms=βms,
                                rms=rms, ndrops=ndrops, density=1., verbose=0)
                if use_teacher_weight_mask
                    set_weight_mask!(g, teacher)
                else
                    set_weight_mask!(g, w)
                end
                initrand!(g)
                fixtopbottom!(g)

                for l=2:g.L+1
                    for k in 1:g.layers[l].K
                        g.layers[l].allhext[k] .= hext[l-1][k] .* g.layers[l].weight_mask[k]
                    end
                end

                e, δ = converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar,
                                    altsolv=altsolv, altconv=altconv, plotinfo=plotinfo,
                                    teacher=teacher, verbose=verbose_in)
                converged += (δ < ϵ)
                solved    += (e == 0)

                for l=2:g.L+1
                    for k in 1:g.layers[l].K
                        @assert all(isfinite, g.layers[l].allh[k])
                        # hext[l-1][k] .= reinfpar.r .* g.layers[l].allh[k] .* g.layers[l].weight_mask[k]
                        hext[l-1][k] .= g.layers[l].allh[k] .* g.layers[l].weight_mask[k]
                    end
                end
                print("b = $b / $num_batches\r")
            end
            w = getW(g)
            E = sum(Int[forward(w, ξ[:, a])[1][1] != σ[a] for a=1:(size(ξ)[2])])

            @printf("Epoch %i (conv=%.2f, solv=%.2f): E=%i r=%.3f rstep=%f\n",
                     epoch, (converged/num_batches), (solved/num_batches), E, reinfpar.r, reinfpar.rstep)
            update_reinforcement!(reinfpar)
            plot_info(g, 0, verbose=verbose, teacher=teacher)
            altsolv && (E==0) && break
        end
    end
    if batchsize > 0
        w = getW(g)
        E = sum(Int[forward(w, ξ[:, a])[1][1] != σ[a] for a=1:(size(ξ)[2])])
        stab = -1
        # g.layers[1] = InputLayer(ξ)
        # g.layers[end] = OutputLayer(σ, β=β)
    else
        E, stab = energy(g)
    end
    return g, getW(g), teacher, E, stab
end

end #module
