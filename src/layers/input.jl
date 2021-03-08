mutable struct InputLayer{T,S} <: AbstractLayer
    l::Int
    Bup::T  # field from fact  ↑ to y
    x::S
end

@functor InputLayer

function InputLayer(x::AbstractMatrix)
    return InputLayer(1, zeros(F, 0,0), x)
end

initrand!(layer::InputLayer) = nothing
