"""
Input from bottom.allpu and top.allpd
and modifies its allpu and allpd
"""
∞ = 10000

# include("utils/Magnetizations.jl")
# using .Magnetizations

abstract type AbstractLayer end
mutable struct DummyLayer <: AbstractLayer
end

include("layers/input.jl")
include("layers/output.jl")
include("layers/maxsum.jl")
include("layers/bp.jl")
include("layers/tap.jl")
include("layers/bpi.jl")
include("layers/parity.jl")
include("layers/bp_real.jl")

istoplayer(layer::AbstractLayer) = (typeof(layer.top_layer) == OutputLayer)
isbottomlayer(layer::AbstractLayer) = (typeof(layer.bottom_layer) == InputLayer)
isonlylayer(layer::AbstractLayer) = istoplayer(layer) && isbottomlayer(layer)

function Base.show(io::IO, layer::L) where {L <: Union{TapExactLayer,TapLayer}}
    @extract layer K N M allm allmy allmh allpu allpd
    println(io, "m=$(allm[1])")
    println(io, "my=$(allmy[1])")
end

getW(lay::AbstractLayer) = getWBinary(lay)
getW(lay::BPRealLayer) = getWReal(lay)
getWReal(lay::AbstractLayer) = lay.allm

function getWBinary(lay::AbstractLayer)
    W = [Float64[1-2signbit(m) for m in magk] for magk in getWReal(lay)] # TODO return BitArray
    if hasproperty(lay, :weight_mask)
        for k in 1:length(W)
            W[k] .*= lay.weight_mask[k]
        end
    end
    return W
end

function energy(lay::OutputLayer, ξ::Vector, a)
    @extract lay: labels
    @assert length(ξ) == 1
    return all((labels[a] .* ξ) .> 0) ? 0 : 1
end

signB(x::T) where {T} = sign(x + 1e-10)


forward(lay::AbstractLayer, ξ::Vector) = forwardBinary(lay, ξ)
forward(lay::BPRealLayer, ξ::Vector) = forwardReal(lay, ξ)
forward(lay::ParityLayer, ξ::Vector) = forwardParity(lay, ξ)

function forward(W::Vector{Vector{T}}, ξ::Vector) where T <: Number
    stability = map(w->dot(ξ, w), W)
    σks = Int[signB(stability[k]) for k=1:length(W)]
    return σks
end

function forward(W::VecVecVec, x::Vector)
    L = length(W)
    for l = 1:L
        x = forward(W[l], x)
    end
    return x
end

function forwardBinary(lay::AbstractLayer, ξ::Vector)
    @extract lay: N K
    W = getWBinary(lay)
    forward(W, ξ)
end

function forwardReal(lay::AbstractLayer, ξ::Vector)
    @extract lay: N
    W = getWReal(lay)
    forward(W, ξ)
end

function forwardParity(lay::AbstractLayer, ξ::Vector)
    @extract lay: N K
    @assert N==2
    @assert K==1
    W = getWReal(lay)
    σks = [sign(ξ[1]*ξ[2])]

    return σks
end

# initYBottom!(lay::AbstractLayer, a::Int) = updateVarY!(lay, a) #TODO define for every layer mutable struct

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
