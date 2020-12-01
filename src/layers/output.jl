mutable struct OutputLayer <: AbstractLayer
    l::Int
    labels::AbstractVector
    B 
    β::Float64
end

function OutputLayer(y::AbstractVector; β=Inf)
    @assert β >= 0.
    @assert all(y.^2 .== 1)
    M = length(y)
    B = β .* reshape(y, 1, :)
    return OutputLayer(-1, y, B, β)
end

initrand!(layer::OutputLayer) = nothing
