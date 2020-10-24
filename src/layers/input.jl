mutable struct InputLayer <: AbstractLayer
    l::Int
    Bup  # field from fact  ↑ to y
    x::Matrix
end

function InputLayer(x::Matrix)
    return InputLayer(1, zeros(0,0), x)
end

initrand!(layer::InputLayer) = nothing
