∞ = 10000

# include("utils/Magnetizations.jl")
# using .Magnetizations

abstract type AbstractLayer end
mutable struct DummyLayer <: AbstractLayer end

include("input.jl")
include("output.jl")
include("bp.jl")
include("bp_exact.jl")
include("tap.jl")
include("tap_exact.jl")
include("bpi.jl")
include("continuous_bpi.jl")
include("mf.jl")
include("argmax.jl")

isfrozen(layer::AbstractLayer) = layer.isfrozen
freeze!(layer::AbstractLayer) = layer.isfrozen = true
unfreeze!(layer::AbstractLayer) = layer.isfrozen = false

isbottomlayer(layer::AbstractLayer) = layer.bottom_layer isa InputLayer

signB(x::T) where {T} = sign(x + 1f-10)

function forward(W::Vector{Vector{T}}, x) where T <: Number
    Wmat =  vcat([w' for w in W]...)
    forward(Wmat, x)
end

function forward(W::AbstractMatrix{T}, x) where T <: Number
    return signB.(W*x)
end

function forward(W::Vector, x)
    L = length(W)
    for l = 1:L
        x = forward(W[l], x)
    end
    return x
end

chain!(lay1::InputLayer, lay2::OutputLayer) = error("Cannot chain InputLayer and OutputLayer")

function chain!(lay1::AbstractLayer, lay2::OutputLayer)
    lay2.l = lay1.l+1
    lay1.top_layer = lay2
end

function chain!(lay1::InputLayer, lay2::AbstractLayer)
    lay2.l = lay1.l+1
    lay2.bottom_layer = lay1
end

function chain!(lay1::AbstractLayer, lay2::AbstractLayer)
    lay2.l = lay1.l+1
    lay1.top_layer = lay2
    lay2.bottom_layer = lay1
end

function set_weight_mask!(lay::AbstractLayer, m::AbstractMatrix)
    lay.weight_mask .= m
end

weight_mean(l::AbstractLayer) = l.m
weight_var(l::AbstractLayer) = l.σ

function compute_overlaps(layer::AbstractLayer; teacher=nothing)
    @extract layer: K N
    # q0 = Float64[]
    # qWαβ = Float64[]
    R = Float64[]

    # for k=1:K
    #     # Nk = hasproperty(layer, :weight_mask) ?
    #     #         sum(layer.weight_mask[k,:]) : N
    #     if hasproperty(layer, :allm)
    #         push!(q0, dot(layer.allm[k], layer.allm[k]) / N)
    #     elseif hasproperty(layer, :m)
    #         push!(q0, dot(layer.m[k,:], layer.m[k,:]) / N)
    #     end
    #     if teacher !== nothing
    #         if hasproperty(layer, :allm)
    #             push!(R, dot(layer.allm[k], teacher[k,:]) / N)
    #         elseif hasproperty(layer, :m)
    #             push!(R, dot(layer.m[k,:], teacher[k,:])/ N)
    #         end
    #     end
    #     for p=k+1:K
    #         # if hasproperty(layer, :weight_mask)
    #         #     Np = sum(layer.weight_mask[p,:])
    #         # else
    #         #     Np = N
    #         # end
    #         if hasproperty(layer, :allm)
    #             push!(qWαβ, dot(layer.allm[k], layer.allm[p])
    #                     / N)
    #         elseif hasproperty(layer, :m)
    #             push!(qWαβ, dot(layer.m[k,:], layer.m[p,:])
    #                     / N)
    #         end
    #         # push!(q, dot(W[l][k],W[l][p])/K[l])
    #         # push!(qWαβ, dot(layer.allm[k], layer.allm[p]) / sqrt(q0[k]*q0[p])/K[l])
    #     end
    # end
    m = weight_mean(layer)
    σ = weight_var(layer)

    @tullio q0[k] := m[k,i] * m[k,i]
    @tullio Δ0[k] := σ[k,i]
    @tullio qWαβ[k,p] := m[k,i] * m[p,i]
    q0 ./= N
    Δ0 ./= N
    qWαβ ./= N
    return q0, Δ0, qWαβ, R
end
