mutable struct OutputLayer{T,S<:AbstractMatrix} <: AbstractLayer
    l::Int
    y::T
    B::S
    A::S
    β::Float32
end

function OutputLayer(y::AbstractVector; β=Inf)
    @assert β >= 0.
    B = β .* reshape(y, 1, :)
    A = fill!(similar(B), β)
    return OutputLayer(-1, y, B, A, Float32(β))
end

@functor OutputLayer

function set_output!(lay::OutputLayer, y)
    lay.B .= lay.β .* reshape(y, 1, :)
    lay.A .= lay.β
    lay.y = y
end

initrand!(layer::OutputLayer) = nothing
