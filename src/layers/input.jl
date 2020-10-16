mutable struct InputLayer <: AbstractLayer
    l::Int
    allpu::VecVec # p(σ=up) from fact ↑ to y
    x::Matrix
end

function InputLayer(x::Matrix)
    return InputLayer(1, VecVec(), x)
end

initrand!(layer::InputLayer) = nothing
