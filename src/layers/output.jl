mutable struct OutputLayer <: AbstractLayer
    l::Int
    labels::IVec
    B 
    β::Float64
end

function OutputLayer(y::Vector{Int}; β=Inf)
    @assert β >= 0.
    @assert all(y -> y ∈ [-1,1], y)
    M = length(y)
    B = [β*y[a] for  k=1:1, a=1:M]
    return OutputLayer(-1, y, B, β)
end

initrand!(layer::OutputLayer) = nothing
