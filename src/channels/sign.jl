@kwdef mutable struct ActSign <: AbstractChannel
    name::Symbol
end

(ch::ActSign)(x::Number) = sign(x)
(ch::ActSign)(x::AbstractArray) = sign.(x)

# TOP LAYER
# function ϕout(ch::ActSign, y, ω, V)
#     logH(-y * ω / V)
# end

# function ∂ω_ϕout(ch::ActSign, y, ω, V)
#     GH(-y * ω / V) * y / V
# end
# function ∂²ω_ϕout(ch::ActSign, y, ω, V)
#     g = ∂ω_ϕout(ch, y, ω, V)
#     -ω / V^2 * g - g^2
# end

# INTERMEDIATE LAYER

function ∂ω_ϕout(ch::ActSign, B, A, ω, V)
    return GH2(B, -ω / √V) / √V
end


function ∂²ω_ϕout(ch::ActSign, B, A, ω, V)
    g = ∂ω_ϕout(ch, B, A, ω, V)
    return -ω / V * g - g^2
end

function ∂B_ϕout(ch::ActSign, B, A, ω, V)
    Btot = B + atanh2Hm1(-ω / √V)
    return tanh(Btot)
end

function ∂²B_ϕout(ch::ActSign, B, A, ω, V)
    Btot = B + atanh2Hm1(-ω / √V)
    return 1 - tanh(Btot)^2
end
