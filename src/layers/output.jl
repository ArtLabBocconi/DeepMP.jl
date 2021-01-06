mutable struct OutputLayer <: AbstractLayer
    l::Int
    y::AbstractVector
    B::AbstractMatrix{F}
    β::F
end

function OutputLayer(y::AbstractVector; β=Inf)
    @assert β >= 0.
    @assert all(y.^2 .== 1)
    M = length(y)
    B = F(β) .* reshape(y, 1, :)
    return OutputLayer(-1, y, B, β)
end

function set_output!(lay::OutputLayer, y)
    lay.B .= lay.β .* reshape(y, 1, :)
    lay.y = y
end

initrand!(layer::OutputLayer) = nothing
