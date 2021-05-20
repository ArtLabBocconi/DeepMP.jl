###########################
#       TAP LAYER
#######################################
mutable struct TapLayer <: AbstractLayer
    l::Int

    K::Int
    N::Int
    M::Int
    ϵinit

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

@functor TapLayer

function TapLayer(K::Int, N::Int, M::Int, ϵinit; density=1, isfrozen=false)
    x̂ = zeros(F, N, M)
    Δ = zeros(F, N, M)
    
    m = zeros(F, K, N)
    σ = zeros(F, K, N)
    
    Bup = zeros(F, K, M)
    B = zeros(F, N, M)
    A = zeros(F, N, M)
    
    H = zeros(F, K, N)
    Hext = zeros(F, K, N)
    
    g = zeros(F, K, M)
    ω = zeros(F, K, M)
    V = zeros(F, K, M)
    
    weight_mask = rand(K, N) .< density
    
    return TapLayer(-1, K, N, M, ϵinit,
            x̂, Δ, m, σ,
            Bup, B, A, 
            H, Hext,
            g, ω, V,
            DummyLayer(), DummyLayer(),
            weight_mask, isfrozen)
end


function update!(layer::TapLayer, reinfpar; mode=:both)
    @extract layer: K N M weight_mask
    @extract layer: x̂ Δ m σ 
    @extract layer: Bup B  A H Hext ω  V g
    @extract layer: bottom_layer top_layer
    @extract reinfpar: r ψ
    Δm = 0.

    if mode == :forw || mode == :both
        ## FORWARD
        if !isbottomlayer(layer)
            bottBup = bottom_layer.Bup
            @tullio x̂[i,a] = tanh(bottBup[i,a] + B[i,a])
            # @tullio x̂new[i,a] := tanh(bottom_layer.Bup[i,a] + B[i,a])
            # x̂ .= ψ .* x̂ .+ (1-ψ) .* x̂new
            Δ .= 1 .- x̂.^2
        end
        
        # V .= σ * x̂.^2 + m.^2 * Δ + σ * Δ .+ 1f-8
        # V[k,a] = σ[k,i] * (x̂.^2)[i,a]
        V .= σ * x̂.^2 + m.^2 * Δ 
        @tullio ω[k,a] = m[k,i] * x̂[i,a]
        @tullio ω[k,a] += - g[k,a] * V[k,a] 
        V .+= σ * Δ .+ 1f-8 
        @tullio Bup[k,a] = atanh2Hm1(-ω[k,a] / √V[k,a]) avx=false
    end

    if mode == :back || mode == :both
        ## BACKWARD 
        Btop = top_layer.B 
        @assert size(Btop) == (K, M)
        @tullio g[k,a] = compute_g(Btop[k,a], ω[k,a], √V[k,a])  avx=false
        @tullio Γ[k,a] := g[k,a] * (ω[k,a] / V[k,a] + g[k,a])

        if !isbottomlayer(layer)
            # A .= (m.^2 .+ σ)' * Γ .- σ' * g.^2
            @tullio A[i,a] = m[k,i]^2 * Γ[k,a]
            @tullio B[i,a] = m[k,i] * g[k,a] - σ[k,i] * Γ[k,a]
            @tullio B[i,a] += x̂[i,a] * A[i,a]
        end

        if !isfrozen(layer) 
            # G = Γ * (x̂.^2 .+ Δ)' .- g.^2 * Δ'
            @tullio G[k,i] := Γ[k,a] * x̂[i,a]^2
            @tullio Hin[k,i] := g[k,a] * x̂[i,a]
            @tullio H[k,i] = Hin[k,i] + m[k,i] * G[k,i] + r*H[k,i] + Hext[k,i]
            @tullio H[k,i] += -Δ[i,a] * Γ[k,a]

            mnew = tanh.(H) .* weight_mask
            Δm = mean(abs.(m .- mnew)) 
            m .= ψ .* m .+ (1-ψ) .* mnew
            σ .= (1 .- m.^2) .* weight_mask    
        end
    end
    
    return Δm
end


function initrand!(layer::TapLayer)
    @extract layer: K N M weight_mask ϵinit
    @extract layer: x̂  Δ m σ 
    @extract layer: B A ω H  V Hext
    H .= ϵinit .* randn!(similar(m)) + Hext
    m .= tanh.(H) .* weight_mask
    σ .= (1 .- m.^2) .* weight_mask
end

function fixY!(layer::L, x::AbstractMatrix) where {L <: Union{TapLayer}}
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
    return sign.(W*x .+ 1f-10)
end

function fixW!(layer::L, w=1.) where {L <: Union{TapLayer}}
    @extract layer: K N M m σ  weight_mask
    m .= w .* weight_mask
    σ .= 0
end