mutable struct BPILayer <: AbstractLayer
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
    
    ω 
    V

    top_layer::AbstractLayer
    bottom_layer::AbstractLayer

    weight_mask
    isfrozen::Bool
end


function BPILayer(K::Int, N::Int, M::Int; density=1., isfrozen=false)
    # for variables W
    x̂ = zeros(N, M)
    Δ = zeros(N, M)
    
    m = zeros(K, N)
    σ = zeros(K, N)
    
    Bup = zeros(K, M)
    B = zeros(N, M)
    A = zeros(N, M)
    
    H = zeros(K, N)
    Hext = zeros(K, N)
    
    ω = zeros(K, M)
    V = zeros(K, M)
    
    weight_mask = [[rand() < density ? 1 : 0 for i=1:N] for i=1:K]

    return BPILayer(-1, K, N, M,
            x̂, Δ, m, σ,
            Bup, B, A, 
            H, Hext,
            ω, V,
            DummyLayer(), DummyLayer(),
            weight_mask, isfrozen)
end

# function compute_g(B, ω, V)
#     1/√V * GH(B, -ω / √V)
# end

function update!(layer::BPILayer, reinfpar)
    @extract layer: K N M
    @extract layer: x̂ Δ m  σ 
    @extract layer: Bup B A H Hext ω  V
    @extract layer: bottom_layer top_layer
    @extract reinfpar: r
    Δm = 0.

    ## FORWARD
    if !isbottomlayer(layer)
        @tullio x̂[i,a] = tanh(bottom_layer.Bup[i,a] + B[i,a])
        Δ .= 1 .- x̂.^2
    end
    
    @tullio ω[k,a] = m[k,i] * x̂[i,a]
    V .= σ * x̂.^2 + m.^2 * Δ + σ * Δ
    @tullio Bup[k,a] = atanh2Hm1(-ω[k,a] / √V[k,a]) avx=false

    @assert all(isfinite, Bup)

    ## BACKWARD 
    Btop = top_layer.B 
    @assert size(Btop) == (K, M)
    @tullio gcav[k,i,a] := compute_g(Btop[k,a], ω[k,a] - m[k,i]*x̂[i,a], V[k,a])  avx=false
    @tullio g[k,a] := compute_g(Btop[k,a], ω[k,a], V[k,a])  avx=false
    # @tullio Γ[k,a] := compute_Γ(Btop[k,a], ω[k,a], V[k,a])
    
    if !isbottomlayer(layer)
        # A .= (m.^2 + σ)' * Γ - σ' * g.^2
        @tullio B[i,a] = m[k,i] * gcav[k,i,a]
        # @tullio B[i,a] = m[k,i] * g[k,a]
    end

    if !isfrozen(layer)
        @tullio Hin[k,i] := gcav[k,i,a] * x̂[i,a]
        # @tullio Hin[k,i] := g[k,a] * x̂[i,a]
        @tullio H[k,i] = Hin[k,i] + r*H[k,i] + Hext[k,i]
        mnew = tanh.(H)
        Δm = maximum(abs, m .- mnew)
        m .= mnew
        σ .= 1 .- m.^2
        @assert all(isfinite, m)
    end
    
    return Δm
end

function initrand!(layer::L) where {L <: Union{BPILayer}}
    @extract layer: K N M
    @extract layer: x̂ Δ m σ 
    @extract layer: B A ω H V
    # TODO reset all variables
    ϵ = 1e-1
    H .= ϵ .* randn(K, N)
    m .= tanh.(H)
    σ .= 1 .- m.^2
end

function fixY!(layer::L, x::Matrix) where {L <: Union{BPILayer}}
    @extract layer: K N M 
    @extract layer: x̂ Δ m σ 
    @assert size(x) == size(x̂)
    x̂ .= x
    Δ .= 0
end

function getW(layer::BPILayer)
    return sign.(layer.m)
end

function forward(layer::BPILayer, x)
    @extract layer: N K
    @assert size(x, 1) == N
    W = getW(layer)
    return sign.(W*x .+ 1e-10)
end

function fixW!(layer::BPILayer, w=1.)
    @extract layer: K N M m σ
    m .= w
    σ .= 0
end

