
abstract type AbstractLayer end

mutable struct DummyLayer <: AbstractLayer end

isfrozen(layer::AbstractLayer) = layer.isfrozen
freeze!(layer::AbstractLayer) = layer.isfrozen = true
unfreeze!(layer::AbstractLayer) = layer.isfrozen = false

isbottomlayer(layer::AbstractLayer) = layer.bottom_layer isa InputLayer
istoplayer(layer::AbstractLayer) = layer.top_layer isa OutputLayer

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
