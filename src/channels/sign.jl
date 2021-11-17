@kwdef mutable struct ActSign <: AbstractChannel
    name::Symbol
end

(ch::ActSign)(x) =  sign(x)

function ϕout(ch::ActSign, y, ω, V)
    logH(-y * ω / V)
end

# top layer
function ∂ω_ϕout(ch::ActSign, y, ω, V)
    GH(-y * ω / V) * y / V
end

# intermediate layer
function ∂ω_ϕout(ch::ActSign, A, B, ω, V)
    GH2(B, -ω / V) / V
end

function ∂²ω_ϕout(ch::ActSign, y, ω, V)
    g = ∂ω_ϕout(ch, y, ω, V)
    -ω / V^2 * g - g^2
end

function ∂²ω_ϕout(ch::ActSign, A, B, ω, V)
    g = ∂ω_ϕout(ch, A, B, ω, V)
    - ω / V^2 * g - g^2
end

# function ∂B_ϕout(ch::ActSign, Atot, Btot)
#     tanh(Btot)
# end

# function ∂B²_ϕout(ch::ActSign, Atot, Btot)
#     1 - tanh(Btot)^2
# end

# function Bout_ϕout(ch::ActSign, ω, V)
#     atanh2Hm1(-ω / V)
# end

function ∂B_ϕout(ch::ActSign, A, B, ω, V)
    Btot = B + atanh2Hm1(-ω / V)
    tanh(Btot)
end

function ∂²B_ϕout(ch::ActSign, A, B, ω, V)
    Btot = B + atanh2Hm1(-ω / V)
    1 - tanh(Btot)^2
end



# function update_ĥσ!(ch::ActSign, ĥ, hnew, σ, A, B, c, d, updmask)
#     D = atanh2Hm1.(.-c ./ sqrt.(d))
#     @. hnew = tanh(B + D)
#     hnew[.!updmask] .= ĥ[.!updmask]
#     @. σ = 1 - hnew^2
#     updmask .&= σ .> 1e-7 # UPDATE the update mask
#     # hnew[.!updmask] .= sign.(ĥ[.!updmask])

#     Δĥ = mean(abs.(hnew .- ĥ))
#     ĥ .= hnew
    
#     @assert all(isfinite.(ĥ))
#     @assert all(isfinite.(σ))

#     return Δĥ
# end

# function update_gdg!(g, ∂g, A, B, c, d, act::ActSign)
#     @. g = GH(B, -c/√(d)) / √(d)
#     @. ∂g = - c/d * g - g.^2
#     @assert all(isfinite.(g))
#     @assert all(isfinite.(∂g))
# end

