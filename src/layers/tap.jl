###########################
#       TAP LAYER
#######################################
mutable struct TapLayer <: AbstractLayer
    l::Int

    K::Int
    N::Int
    M::Int

    x̂ 
    Δ

    m 
    σ 

    Bup
    B 
    A 
    
    H
    Hext
    
    g 
    ω 
    V
    
    top_layer::AbstractLayer
    bottom_layer::AbstractLayer

    weight_mask
    isfrozen::Bool
end

function TapLayer(K::Int, N::Int, M::Int; density=1, isfrozen=false)
    x̂ = zeros(N, M)
    Δ = zeros(N, M)
    
    m = zeros(K, N)
    σ = zeros(K, N)
    
    Bup = zeros(K, M)
    B = zeros(N, M)
    A = zeros(N, M)
    
    H = zeros(K, N)
    Hext = zeros(K, N)
    
    g = zeros(K, M)
    ω = zeros(K, M)
    V = zeros(K, M)
    
    weight_mask = rand(K, N) .< density
    
    return TapLayer(-1, K, N, M,
            x̂, Δ, m, σ,
            Bup, B, A, 
            H, Hext,
            g, ω, V,
            DummyLayer(), DummyLayer(),
            weight_mask, isfrozen)
end


function update!(layer::TapLayer, reinfpar)
    @extract layer: K N M weight_mask
    @extract layer: x̂ Δ m σ 
    @extract layer: Bup B  A H Hext ω  V g
    @extract layer: bottom_layer top_layer
    @extract reinfpar: r
    Δm = 0.

    ## FORWARD
    if !isbottomlayer(layer)
        @tullio x̂[i,a] = tanh(bottom_layer.Bup[i,a] + B[i,a])
        Δ .= 1 .- x̂.^2
    end
    
    # V .= σ * x̂.^2 + m.^2 * Δ + σ * Δ .+ 1e-8
    V .= σ * x̂.^2 + m.^2 * Δ 
    @tullio ω[k,a] = m[k,i] * x̂[i,a]
    @tullio ω[k,a] += - g[k,a] * V[k,a] 
    V .+= σ * Δ .+ 1e-8 
    @tullio Bup[k,a] = atanh2Hm1(-ω[k,a] / √V[k,a]) avx=false
    

    ## BACKWARD 
    Btop = top_layer.B 
    @assert size(Btop) == (K, M)
    @tullio g[k,a] = compute_g(Btop[k,a], ω[k,a], V[k,a])  avx=false
    @tullio Γ[k,a] := g[k,a] * (ω[k,a] / V[k,a] + g[k,a])

    if !isbottomlayer(layer)
        # A .= (m.^2 .+ σ)' * Γ .- σ' * g.^2
        @tullio A[i,a] = m[k,i]^2 * Γ[k,a]
        @tullio B[i,a] = m[k,i] * g[k,a]
        @tullio B[i,a] += x̂[i,a] * A[i,a]
        @tullio B[i,a] += -σ[k,i] * Γ[k,a] #
    end

    if !isfrozen(layer) 
        # G = Γ * (x̂.^2 .+ Δ)' .- g.^2 * Δ'
        @tullio G[k,i] := Γ[k,a] * x̂[i,a]^2
        @tullio Hin[k,i] := g[k,a] * x̂[i,a]
        @tullio H[k,i] = Hin[k,i] + m[k,i] * G[k,i] + r*H[k,i] + Hext[k,i]
        @tullio H[k,i] += -Δ[i,a] * Γ[k,a]

        mnew = tanh.(H) .* weight_mask
        Δm = maximum(abs, m .- mnew) 
        m .= mnew
        σ .= (1 .- m.^2) .* weight_mask    
    end
    
    return Δm
end


function initrand!(layer::TapLayer)
    @extract layer: K N M weight_mask
    @extract layer: x̂  Δ m σ 
    @extract layer: B A ω H  V Hext
    
    ϵ = 1e-1
    H .= ϵ .* randn(K, N) + Hext
    m .= tanh.(H) .* weight_mask
    σ .= (1 .- m.^2) .* weight_mask
end

function fixY!(layer::L, x::Matrix) where {L <: Union{TapLayer}}
    @extract layer: K N M 
    @extract layer: x̂ Δ m  σ 
    @assert size(x) == size(x̂)
    x̂ .= x
    Δ .= 0
end

function getW(layer::L) where L <: Union{TapLayer}
    return sign.(layer.m) .* layer.weight_mask
end

function forward(layer::L, x) where L <: Union{TapLayer}
    @extract layer: N K
    @assert size(x, 1) == N
    W = getW(layer)
    return sign.(W*x .+ 1e-10)
end

function fixW!(layer::L, w=1.) where {L <: Union{TapLayer}}
    @extract layer: K N M m σ  weight_mask
    m .= w .* weight_mask
    σ .= 0
end