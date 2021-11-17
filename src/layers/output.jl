mutable struct OutputLayer{T,S,W} <: AbstractLayer
    l::Int
    y::T
    A::S
    B::S
    β::W
end

function OutputLayer(y::AbstractVector; β=Inf)
    @assert β >= 0.
    B = β .* reshape(y, 1, :)
    A = fill!(similar(B), Inf)
    return OutputLayer(-1, y, A, B, β)
end

@functor OutputLayer

function set_output!(lay::OutputLayer, y)
    lay.B .= lay.β .* reshape(y, 1, :)
    lay.A = fill!(lay.A, Inf)
    lay.y = y
end

initrand!(layer::OutputLayer) = nothing
