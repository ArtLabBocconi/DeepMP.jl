@kwdef mutable struct ActSign <: AbstractChannel
    name::Symbol
end

(ch::ActSign)(x) =  sign(x)

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

function ∂ω_ϕout(ch::ActSign, A, B, ω, V)
    GH2(B, -ω / V) / V
end


function ∂²ω_ϕout(ch::ActSign, A, B, ω, V)
    g = ∂ω_ϕout(ch, A, B, ω, V)
    -ω / V^2 * g - g^2
end

function ∂B_ϕout(ch::ActSign, A, B, ω, V)
    Btot = B + atanh2Hm1(-ω / V)
    tanh(Btot)
end

function ∂²B_ϕout(ch::ActSign, A, B, ω, V)
    Btot = B + atanh2Hm1(-ω / V)
    1 - tanh(Btot)^2
end
