"""
Input from bottom.allpu and top.allpd
and modifies its allpu and allpd
"""
∞ = 10000

# include("utils/Magnetizations.jl")
# using .Magnetizations

abstract type AbstractLayer end
mutable struct DummyLayer <: AbstractLayer end

include("input.jl")
include("output.jl")
include("maxsum.jl")
include("bp.jl")
include("tap.jl")
include("bpi.jl")
include("parity.jl")
include("bp_real.jl")
include("bp2.jl")

isfrozen(layer::AbstractLayer) = layer.isfrozen
freeze!(layer::AbstractLayer) = layer.isfrozen = true
unfreeze!(layer::AbstractLayer) = layer.isfrozen = false

isbottomlayer(layer::AbstractLayer) = (typeof(layer.bottom_layer) == InputLayer)

signB(x::T) where {T} = sign(x + 1e-10)

function forward(W::Vector{Vector{T}}, x) where T <: Number
    Wmat =  vcat([w' for w in W]...)
    forward(Wmat, x)
end

function forward(W::Matrix{T}, x) where T <: Number
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
    lay1.top_allpd = lay2.allpd
    lay2.l = lay1.l+1
    lay1.top_layer = lay2
end

function chain!(lay1::InputLayer, lay2::AbstractLayer)
    lay2.l = lay1.l+1
    lay2.bottom_allpu = lay1.allpu
    lay2.bottom_layer = lay1
    for a=1:lay2.M
        initYBottom!(lay2, a)
    end
end

function chain!(lay1::BPLayer2, lay2::OutputLayer)
    lay2.l = lay1.l+1
    lay1.top_layer = lay2
end

function chain!(lay1::InputLayer, lay2::BPLayer2)
    lay2.l = lay1.l+1
    lay2.bottom_layer = lay1
end

function chain!(lay1::AbstractLayer, lay2::AbstractLayer)
    lay2.l = lay1.l+1
    lay1.top_allpd = lay2.allpd
    lay2.bottom_allpu = lay1.allpu
    lay1.top_layer = lay2
    lay2.bottom_layer = lay1
end

function set_weight_mask!(lay::AbstractLayer, m)
    @assert length(lay.weight_mask) == length(m)
    for k=1:length(m)
        lay.weight_mask[k] .= m[k]
    end
end

function compute_overlaps(layer::AbstractLayer; teacher=nothing)
    @extract layer: K
    q0 = Float64[]
    qWαβ = Float64[]
    R = Float64[]

    for k=1:K
        Nk = hasproperty(layer, :weight_mask) ?
                sum(layer.weight_mask[k]) : K
        if hasproperty(layer, :allm)
            push!(q0, dot(layer.allm[k], layer.allm[k]) / Nk)
        elseif hasproperty(layer, :m)
            push!(q0, dot(layer.m[k,:], layer.m[k,:]) / Nk)
        end
        if teacher !== nothing
            push!(R, dot(layer.allm[k], teacher[k]) / Nk)
        end
        for p=k+1:K
            if hasproperty(layer, :weight_mask)
                Np = sum(layer.weight_mask[p])
            else
                Np = K
            end
            if hasproperty(layer, :allm)
                push!(qWαβ, dot(layer.allm[k], layer.allm[p])
                        / sqrt(Nk*Np))
            elseif hasproperty(layer, :m)
                push!(qWαβ, dot(layer.m[k], layer.m[p])
                        / sqrt(Nk*Np))
            end
            # push!(q, dot(W[l][k],W[l][p])/K[l])
            # push!(qWαβ, dot(layer.allm[k], layer.allm[p]) / sqrt(q0[k]*q0[p])/K[l])
        end
    end
    q0, qWαβ, R
end
