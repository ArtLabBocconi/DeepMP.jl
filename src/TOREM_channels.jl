using Base: @kwdef

## type hierarchy for input and output channels
##### CHANNELS  ##################
abstract type AbstractChannel end

∂ω_ϕout(y, ω, V, chout::AbstractChannel) = deriv(ω->ϕout(y, ω, V, chout),ω)
∂²ω_ϕout(y, ω, V, chout::AbstractChannel) = deriv(ω->∂ω_ϕout(y, ω, V, chout),ω)
∂B_ϕin(B, A, chin::AbstractChannel) = deriv(B->ϕin(B, A, chin), B)
∂²B_ϕin(B, A, chin::AbstractChannel) = deriv(B->∂B_ϕin(B, A, chin), B)

# function ϕ1out(y, ω, V₀, V₁, T, s, chout::AbstractChannel)
#     ∫Dexp(z -> s * ϕout(y, ω + √V₀ * z, V₁, chout))
# end

∂ω_ϕ1out(y, ω, V₀, V₁, T, s, chout::AbstractChannel) = deriv(ω->ϕ1out(y, ω, V₀, V₁, T, s, chout), ω)
∂²ω_ϕ1out(y, ω, V₀, V₁, T, s, chout::AbstractChannel) = deriv(ω->∂ω_ϕ1out(y, ω, V₀, V₁, T, s, chout), ω)
∂V₀_ϕ1out(y, ω, V₀, V₁, T, s, chout::AbstractChannel) = deriv(V₀->ϕ1out(y, ω, V₀, V₁, T, s, chout), V₀)
∂V₁_ϕ1out(y, ω, V₀, V₁, T, s, chout::AbstractChannel) = deriv(V₁->ϕ1out(y, ω, V₀, V₁, T, s, chout), V₁)
∂s_ϕ1out(y, ω, V₀, V₁, T, s, chout::AbstractChannel) = deriv(s->ϕ1out(y, ω, V₀, V₁, T, s, chout), s)
∂B_ϕ1in(B, A₀, A₁, T, s, chin::AbstractChannel) = deriv(B->ϕ1in(B, A₀, A₁, T, s, chin), B)
∂²B_ϕ1in(B, A₀, A₁, T, s, chin::AbstractChannel) = deriv(B->∂B_ϕ1in(B, A₀, A₁, T, s, chin), B)
∂A₀_ϕ1in(B, A₀, A₁, T, s, chin::AbstractChannel) = deriv(A₀->ϕ1in(B, A₀, A₁, T, s, chin), A₀)
∂A₁_ϕ1in(B, A₀, A₁, T, s, chin::AbstractChannel) = deriv(A₁->ϕ1in(B, A₀, A₁, T, s, chin), A₁)

##### OUTPUT CHANNELS #########
# Δ is the variance of additive gaussian PRE-ACTIVATION noise
# T for CONTINUOUS observations is the variance of additive Gaussian POST-Activation noise

#TODO 


#################################

@kwdef mutable struct ActAbs <: AbstractChannel
    name::Symbol
    Δ::Float64 = 0
    T::Float64 = 0
end

(ch::ActAbs)(x) = abs(x + √(ch.Δ) * randn())

function ϕout(y, ω, V, chout::ActAbs)
    @extract chout: Δ T
    @assert T == 0  #TODO Implement temperature

    V += Δ
    # a1 = -0.5 * (ω-y)^2 / (V+Δ)
    # a2 = -0.5 * (ω+y)^2 / (V+Δ)
    # am = max(a1, a2)
    # res = am + log(exp(a1-am) + exp(a2-am)) - 0.5*log(2π*(V+Δ))
    # # @show y
    # res

    -0.5*(ω^2 + y^2) / V + log(2cosh(y*ω/V)) - 0.5*log(2π*V)
end

function  ∂ω_ϕout(y, ω, V, chout::ActAbs)
    @extract chout: Δ T
    @assert T == 0  #TODO Implement temperature

    V += Δ
    # kappa = 1e-10
    # ifelse(y < kappa,
    #     (y - ω) / V,
    #     (-ω + y*tanh(y*ω/V)) / V
    (-ω + y*tanh(y*ω/V)) / V
end

function  ∂²ω_ϕout(y, ω, V, chout::ActAbs)
    @extract chout: Δ T
    @assert T == 0  #TODO Implement temperature
    V += Δ
    -1 / V + (1 - tanh(y*ω/V)^2)*(y/V)^2
end

# TODO use logH for numerical stability
function ϕ1out(y, ω, V₀, V₁, T, s, chout::ActAbs)
    @extract chout: Δ
    @assert T == 0  #TODO Implement temperature

    a = sqrt((1 + 2*V₁)*(1 + 2*s*V₀ + 2*V₁))
    # @assert abs(-((2*s*V₀*y - ω*(1 + 2*V₁))/(sqrt(V₀)*a))) < 45
    Z1 = H(-((2*s*V₀*y - ω*(1 + 2*V₁))/(sqrt(V₀)*a))) * exp(-(s*(ω + y)^2)/(1 + 2*s*V₀ + 2*V₁))
    Z2 = H(-((2*s*V₀*y + ω*(1 + 2*V₁))/(sqrt(V₀)*a)))*exp(-(s*(ω - y)^2)/(1 + 2*s*V₀ + 2*V₁))
    Z = (1 + 2*V₁)*(Z1 + Z2) / a

    1/s * log(Z)
end

####

@kwdef mutable struct ActId <: AbstractChannel
    name::Symbol
    Δ::Float64 = 0
    T::Float64 = 0  # For linear channels T and Δ are equivalent
end

(ch::ActId)(x) = x + √(ch.Δ) * randn() + √(ch.T) * randn()

function ϕout(y, ω, V, chout::ActId)
    @extract chout: Δ T
    V += Δ + T
    -0.5 * ((ω-y)^2/V + log(2π*V)) # + log(T)
end

function ϕ1out(y, ω, V₀, V₁, T, s, chout::ActId, noise_on_replicas=true)
    @extract chout: Δ
    # @assert T == 0
    # TODO check Δ  and  T dependence
    if noise_on_replicas
        V₁ += Δ
    else
        V₀ += Δ
    end

    if T > 0
        ϕ = -(ω-y)^2 / (T + s*V₀ + V₁)
        ϕ += - 1/s*(s-1)*log(1 + V₁/T)
        ϕ += - 1/s*log(1 + (V₁ + s*V₀)/T)

    elseif T==0 && s < Inf
        # ϕ = 0.5*(-(ω-y)^2/(V₀ + (1 + V₁)/s) - log((V₀ +(1+V₁)/s) /(1+V₁)))
        ϕ = -(ω-y)^2/(s*V₀ + 1 + V₁) - 1/s*log((s*V₀ + 1 + V₁) /(1+ V₁))

    else # T==0 && s == Inf
        ϕ = -(ω-y)^2/V₀ - log(V₀/(1+V₁))
    end

    #ϕ += -2log(s)) # remove this for use at s=Inf
    ϕ/2
end


## IN CHANNELS #########

@kwdef mutable struct RegLp <: AbstractChannel
    name::Symbol
    p::Float64
    λ::Float64 = 1
end

(ch::RegLp)(x) = ch.λ * abs(x)^ch.p

####

#TODO change name to VarSelUninf
"""
This represents the Variable Selection formulation
of compressed
"""
@kwdef mutable struct RegL0 <: AbstractChannel
    name::Symbol
    λ::Float64 = 1
    ρ::Float64 = Inf  # set this to finite number to hard constrain L0 norm instead of soft contraining with lambda
    T::Float64 = 0    # temperature used in inference phase
end

(ch::RegL0)(x) = ch.λ * (x != 0) # cannot be used as a prior

function ϕ1in(B, A₀, A₁, T, s, ch::RegL0)
    @extract ch: λ
    if T > 0
        a = -λ + 0.5*s*B^2/(A₁-s*A₀) + 0.5*log(A₁/(A₁-s*A₀))
        a += 0.5*s*log(2π / A₁)
        mx = max(0, a)
        return (mx + log(exp(-mx) + exp(a-mx))) / s

    elseif T == 0  && s < Inf
            a = -λ + 0.5*s*B^2/(A₁-s*A₀) + 0.5*log(A₁/(A₁-s*A₀))
            mx = max(0, a)
            return (mx + log(exp(-mx) + exp(a-mx))) / s

    else # T == 0  && s == Inf
        a = -λ + 0.5*B^2/A₁ + 0.5*log(A₀/A₁)
        mx = max(0, a)
        return mx + log(exp(-mx) + exp(a-mx))
    end
end

∂λ_ϕ1in(B, A₀, A₁, T, s, ch::RegL0) = grad(ch -> ϕ1in(B, A₀, A₁, T, s, ch), ch).λ

######

@kwdef mutable struct RegL1 <: AbstractChannel
    name::Symbol
    λ::Float64 = 1
end

(ch::RegL1)(x) = ch.λ * abs(x)

ϕin(B, A, ch::RegL1) = abs(B) > ch.λ ? (abs(B)-ch.λ)^2/(2A) : 0.
∂B_ϕin(B, A, ch::RegL1) = B > ch.λ  ? (B-ch.λ)/A :
                          B < -ch.λ ? (B+ch.λ)/A : 0.
∂²B_ϕin(B, A, ch::RegL1) = abs(B) > ch.λ ? 1/A : 0.
∂λ_ϕin(B, A, ch::RegL1) = deriv(ch -> ϕin(B, A, ch), ch).λ

####

@kwdef mutable struct PriorGauss <: AbstractChannel
    name::Symbol
    σ²::Float64 = 1    # variance
end

(ch::PriorGauss)() = sqrt(ch.σ²) * randn()

function ϕin(B, A, chin::PriorGauss)
    @extract chin: σ²
    1/2*(B^2/(1/σ² + A)) -1/2*log(σ²)- 1/2*log(1/σ² + A)
end


function ϕ1in(B, A₀, A₁, T, s, chin::PriorGauss)
    @extract chin: σ² #T
    @assert T == 0 # TODO implements finite T
    1/2 * (B^2/(A₁ + 1/σ² - A₀*s) + log(A₁+1/σ²)/s - log(A₁ + 1/σ² - A₀*s)/s)
end

#####

@kwdef mutable struct PriorRadam <: AbstractChannel
    name::Symbol
end

(ch::PriorRadam)() = rand([-1, 1])

ϕin(B, A, chin::PriorRadam) = -A/2 + logcosh(B)
∂B_ϕin(B, A, chin::PriorRadam) = tanh(B)
# ∂²B_ϕin(B, A, chin::PriorRadam) = max(1e-8, 1-tanh(B)^2) # for numerical stability

#####

@kwdef mutable struct PriorBernGauss <: AbstractChannel
    name::Symbol
    ρ::Float64          # non-zero density
    σ²::Float64 = 1     # variance
end

(ch::PriorBernGauss)() = rand() < ch.ρ ? ch.σ² * randn() : 0.

function ϕin(B, A, chin::PriorBernGauss)
    @extract chin: ρ σ²
    g1 = log(1-ρ)
    g2 = log(ρ) + 1/2*(B^2/(1/σ² + A)) -1/2*log(σ²)- 1/2*log(1/σ² + A)
    gm = max(g1, g2)
    return gm + log(exp(g1-gm) + exp(g2-gm))
end

#####

@kwdef mutable struct PriorBernRadam <: AbstractChannel
    name::Symbol
    ρ::Float64       # non-zero density
end
(ch::PriorBernRadam)() = rand() < ch.ρ ? rand([-1, 1]) : 0

function ϕin(B, A, chin::PriorBernRadam)
    @extract chin: ρ
    g1 = log(1-ρ)
    g2 = log(ρ) - 1/2*A + logcosh(B)
    gm = max(g1, g2)
    return gm + log(exp(g1-gm) + exp(g2-gm))
end


function update_ĥσ!(ĥ, hnew, σ, A, B, prior::PriorBernGauss, updmask)
    @. hnew = ∂B_ϕin(B, A, prior)
    @. σ = ∂²B_ϕin(B, A, prior)
    
    Δĥ = mean(abs.(hnew .- ĥ))
    ĥ .= hnew
    
    @assert all(isfinite.(ĥ))
    @assert all(isfinite.(σ))

    return Δĥ
end

function update_ĥσ!(ĥ, hnew, σ, A, B, prior::PriorRadam, updmask)
    @. hnew = tanh(B)
    hnew[.!updmask] .= ĥ[.!updmask]
    @. σ = 1 - hnew^2
    updmask .&= σ .> 1e-7 # UPDATE the update mask
    # hnew[.!updmask] .= sign.(ĥ[.!updmask])
    
    Δĥ = mean(abs.(hnew .- ĥ))
    ĥ .= hnew
    
    @assert all(isfinite.(ĥ))
    @assert all(isfinite.(σ))

    return Δĥ
end

function update_ĥσ!(ĥ, hnew, σ, A, B, prior::PriorGauss, updmask)
    μ = 0.  # should be incorporated in the prior
    if all(prior.σ² .== 0)
        @. hnew = μ
        @. σ = 0
    else
        @. hnew = (B + μ / prior.σ²) / (A + 1 / prior.σ²)
        @. σ = 1 / (A + 1/prior.σ²)
    end
    Δĥ = mean(abs.(hnew .- ĥ))
    ĥ .= hnew
    # ψ = 0.
    # @. ĥ = clip(ψ * ĥ + (1 - ψ) * (hnew - ĥ))

    @assert all(isfinite.(ĥ))
    @assert all(isfinite.(σ))

    return Δĥ
end

function update_ĥσ!(ĥ, hnew, σ, A, B, c, d, act::ActId, updmask)
    @. hnew = (B + c / d) / (A + 1 / d)

    Δĥ = mean(abs.(hnew .- ĥ))
    # ψ = 0.2
    # @. ĥ = ψ*ĥ + (1-ψ)*hnew
    @. ĥ = hnew

    # @. σ = ψ * σ + (1-ψ) * 1 / (A + C)
    @. σ = 1 / (A + 1 / d)
    @assert all(σ .>= 0)

    return Δĥ
end


function update_gdg!(g, ∂g, A, B, c, d, act::ActId)
    if any(A .== Inf)
        @assert all(A .== Inf)
        @. g = (B - c) / d
    else
        @. g = (B/A - c) / (d + 1/A)
    end
    @. ∂g = -1 / (d + 1/A)
    @assert all(isfinite.(g))
    @assert all(isfinite.(∂g))
end

