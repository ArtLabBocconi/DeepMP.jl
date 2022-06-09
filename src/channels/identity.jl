#### ACT RELU   ###############

## type hierarchy for input and output channels


@kwdef mutable struct ActIdentity <: AbstractChannel
    name::Symbol
end

(ch::ActIdentity)(x) =  x


function ϕout(ch::ActIdentity, B, A, ω, V)
    y = B / A
    invVA = 1 / (V * A) 
    res = -ω^2 / V + (y + ω * invVA)^2 / (1 + invVA / A) - log1p(invVA) # - log(A) + log(2π) 
    return res / 2 
end

# INTERMEDIATE LAYER

function ∂ω_ϕout(ch::ActIdentity, B, A, ω, V)
    y = B / A
    invVA = 1 / (V * A)  
    return (-ω + (y + ω * invVA) / (1 + invVA)) / V
end

function ∂²ω_ϕout(ch::ActIdentity, B, A, ω, V)
    return (-1 + invVA / (1 + invVA)) / V
end

function ∂B_ϕout(ch::ActIdentity, B, A, ω, V)
    y = B / A
    invVA = 1 / (V * A)  
    return (y + ω * invVA) / (1 + invVA)
end

function ∂²B_ϕout(ch::ActIdentity, B, A, ω, V)
    return 1 / (A + 1 / V)
end
