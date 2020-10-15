mutable struct InputLayer <: AbstractLayer
    l::Int
    allpu::VecVec # p(σ=up) from fact ↑ to y
    ξ::Matrix
end

function InputLayer(ξ::Matrix)
    return InputLayer(1, VecVec(), ξ)
end

initrand!(layer::InputLayer) = nothing
