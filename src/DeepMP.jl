module DeepMP

using ExtractMacro
using FastGaussQuadrature
using SpecialFunctions
using Printf
using Random
using LinearAlgebra
using Statistics

using PyPlot

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
                                , altsolv::Bool=false, altconv = false, plotinfo=-1
                                , reinfpar::ReinfParams=ReinfParams()
                                , verbose::Int=1)

    for it=1:maxiters
        Δ = update!(g, reinfpar)

        E, h = energy(g)
        verbose > 0 && @printf("it=%d \t r=%.3f ry=%.3f \t E=%d \t Δ=%f \n"
                , it, reinfpar.r, reinfpar.ry, E, Δ)
        # println(h)
        plotinfo >=0  && plot_info(g, plotinfo, verbose=verbose)
        update_reinforcement!(reinfpar)
        if altsolv && E == 0
            println("Found Solution: correctly classified $(g.M) patterns.")
            break
        end
        if altconv && Δ < ϵ
            println("Converged!")
            break
        end
    end
end

function rand_teacher(K::Vector{Int}; density=1.)
    L = length(K)-1
    @assert K[L+1] == 1

    if isa(density, Number)
        density = fill(density, L)
        density[L] = 1
    end
    @assert density[L] == 1
    @assert length(density) == L

    T = Float64
    W = Vector{Vector{Vector{T}}}()
    for l=1:L
        push!(W, T[rand([-1,1], K[l]) for k=1:K[l+1]])    
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

function solve(; K::Vector{Int} = [101,3], α::Float64=0.6,
                 seedξ::Int=-1, realξ = false,
                 dξ::Vector{Float64} = Float64[], nξ::Vector{Int} = Int[],
                 maketree = false, kw...)

    seedξ > 0 && Random.seed!(seedξ)
    numW = length(K)==2 ? K[1]*K[2]  : sum(l->K[l]*K[l+1],1:length(K)-2)
    maketree && (numW = div(numW, K[2]))
    N = K[1]
    ξ = zeros(K[1], 1)

    if length(nξ) == 0
        M = round(Int, α * numW)
        if realξ
            ξ = randn(K[1], M)
        else
            ξ = rand([-1.,1.], K[1], M)
        end
        # σ = ones(Int, M)
        σ = rand([-1,1], M)
    else
        ξ0 = rand([-1.,1.], K[1],1)
        nξ[end] = round(Int, α * numW / prod(nξ[1:end-1]))
        M = round(Int, prod(nξ))
        @assert all(dξ[1:end-1] .>= dξ[2:end])
        for l=1:length(nξ)
            nb = size(ξ0, 2)
            na = nξ[l]
            d = dξ[l]
            @assert 0 <= d <= 0.5
            pflip = 1-sqrt(1-2d)
            ξ = zeros(N, na*nb)
            for a=1:na, b=1:nb
                m = a + (b-1)*na
                for i=1:N
                    ξ[i, m] = rand() < pflip ? rand([-1.,1.]) : ξ0[i,b]
                end
            end
            ξ0 = ξ
        end
        ξ = ξ0
        σ = rand([-1,1], M)
    end
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

function solve(ξ::Matrix, σ::Vector{Int}; maxiters::Int = 10000, ϵ::Float64 = 1e-4,
                K::Vector{Int} = [101, 3, 1],layers=[:tap,:tapex,:tapex],
                r::Float64 = 0., rstep::Float64= 0.001,
                ry::Float64 = 0., rystep::Float64= 0.0,
                ψ = 0., # dumping coefficient
                y = 0, # focusing
                teacher::Union{VecVecVec, Nothing} = nothing,
                altsolv::Bool = true, altconv::Bool = false,
                seed::Int = -1, plotinfo=0,
                β=Inf, βms = 1., rms = 1., ndrops = 0, maketree=false,
                density = 1., # density of fully connected layer
                verbose::Int = 1)

    seed > 0 && Random.seed!(seed)
    g = FactorGraph(ξ, σ, K, layers, β=β, βms=βms, rms=rms, ndrops=ndrops, density=density)
    initrand!(g)
    fixtopbottom!(g)
    maketree && maketree!(g.layers[2])
    reinfpar = ReinfParams(r, rstep, ry, rystep, y, ψ)

    converge!(g, maxiters=maxiters, ϵ=ϵ, reinfpar=reinfpar,
              altsolv=altsolv, altconv=altconv, plotinfo=plotinfo,
              verbose=verbose)

    E, stab = energy(g)
    return g, getW(g), teacher, E, stab
end

end #module
