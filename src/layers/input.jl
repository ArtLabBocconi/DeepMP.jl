mutable struct InputLayer <: AbstractLayer
    l::Int
    Bup  # field from fact  ↑ to y
    x::AbstractMatrix
end

function InputLayer(x::AbstractMatrix)
    return InputLayer(1, zeros(0,0), x)
end

initrand!(layer::InputLayer) = nothing
