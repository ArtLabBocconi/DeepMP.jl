mutable struct OutputLayer <: AbstractLayer
    l::Int
    labels::IVec
    allpd::VecVec # p(σ=up) from fact ↑ to y
    β::Float64
end

function OutputLayer(y::Vector{Int}; β=Inf)
    @assert β >= 0.
    allpd = VecVec()
    K = maximum(y)
    if K<=1 #binary classification
        push!(allpd, Float64[β*y[a] for a=1:length(y)])
        out = OutputLayer(-1, y, allpd, β)
    elseif K >= 2 # K-ary classification
        for k=1:K
            push!(allpd, Float64[y[a]==k ? β : -β for a=1:length(y)])
            out = OutputLayer(-1, y, allpd, β)
        end
    end

    return out
end

initrand!(layer::OutputLayer) = nothing
