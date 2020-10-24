
mutable struct BPLayer <: AbstractLayer
    l::Int
    K::Int
    N::Int
    M::Int

    x̂ 
    x̂cav 
    Δ

    m 
    mcav 
    σ 

    Bup
    B 
    Bcav 
    A 
    
    H
    Hext
    Hcav 

    ω 
    ωcav 
    V

    top_layer::AbstractLayer
    bottom_layer::AbstractLayer

    weight_mask
    isfrozen::Bool
end


function BPLayer(K::Int, N::Int, M::Int; density=1., isfrozen=false)
    # for variables W
    x̂ = zeros(N, M)
    x̂cav = zeros(K, N, M)
    Δ = zeros(N, M)
    
    m = zeros(K, N)
    mcav = zeros(K, N, M)
    σ = zeros(K, N)
    
    Bup = zeros(K, M)
    B = zeros(N, M)
    Bcav = zeros(K, N, M)
    A = zeros(N, M)
    
    H = zeros(K, N)
    Hext = zeros(K, N)
    Hcav = zeros(K, N, M)
    
    ω = zeros(K, M)
    ωcav = zeros(K, N, M)
    V = zeros(K, M)
    
    weight_mask = [[rand() < density ? 1 : 0 for i=1:N] for i=1:K]

    return BPLayer(-1, K, N, M,
            x̂, x̂cav, Δ, m, mcav, σ,
            Bup, B, Bcav, A, 
            H, Hext, Hcav,
            ω, ωcav, V,
            DummyLayer(), DummyLayer(),
            weight_mask, isfrozen)
end


function get_AB(layer::AbstractLayer)
    return 0, layer.B 
end

function compute_g(B, ω, V)
    1/√V * GH(B, -ω / √V)
end

# function  compute_x(lay::BPLayer, B, i, a)
#     tanh(Bup + B)
# end

function update!(layer::BPLayer, reinfpar)
    @extract layer: K N M
    @extract layer: x̂ x̂cav Δ m mcav σ 
    @extract layer: Bup B Bcav A H Hext Hcav ω ωcav V
    @extract layer: bottom_layer top_layer
    @extract reinfpar: r
    
    ## FORWARD
    if !isbottomlayer(layer)
        @tullio x̂cav[k,i,a] = tanh(bottom_layer.Bup[i,a] + Bcav[k,i,a])
        @tullio x̂[i,a] = tanh(bottom_layer.Bup[i,a] + B[i,a])
        Δ .= 1 .- x̂.^2
    end
    
    @tullio ω[k,a] = mcav[k,i,a] * x̂cav[k,i,a]
    @tullio ωcav[k,i,a] = ω[k,a] - mcav[k,i,a] * x̂cav[k,i,a]
    V .= σ * x̂.^2 + m.^2 * Δ + σ * Δ
    @tullio Bup[k,a] = atanh2Hm1(-ω[k,a] / √V[k,a]) avx=false
    

    ## BACKWARD 
    Atop, Btop = get_AB(top_layer)
    @assert size(Btop) == (K, M)
    @tullio gcav[k,i,a] := compute_g(Btop[k,a], ωcav[k,i,a], V[k,a])  avx=false
    @tullio g[k,a] := compute_g(Btop[k,a], ω[k,a], V[k,a])  avx=false
    # @tullio Γ[k,a] := compute_Γ(Btop[k,a], ω[k,a], V[k,a])
    
    if !isbottomlayer(layer)
        # A .= (m.^2 + σ)' * Γ - σ' * g.^2
        @tullio B[i,a] = mcav[k,i,a] * gcav[k,i,a]
        @tullio Bcav[k,i,a] = B[i,a] - mcav[k,i,a] * gcav[k,i,a]
    end

    @tullio Hin[k,i] := gcav[k,i,a] * x̂cav[k,i,a]
    @tullio H[k,i] = Hin[k,i] + r*H[k,i] + Hext[k,i]
    @tullio Hcav[k,i,a] = H[k,i] - gcav[k,i,a] * x̂cav[k,i,a]
    mcav .= tanh.(Hcav)
    mnew = tanh.(H)
    Δm = maximum(abs, m .- mnew) 
    m .= mnew
    σ .= 1 .- m.^2    
    
    return Δm
end

function initrand!(layer::L) where {L <: Union{BPLayer}}
    @extract layer: K N M
    @extract layer: x̂ x̂cav Δ m mcav σ 
    @extract layer: B Bcav A ω H Hcav ωcav V
    # TODO reset all variables
    ϵ = 1e-10
    H .= ϵ .* randn(K, N)
    m .= tanh.(H)
    mcav .= m
    σ .= 1 .- m.^2
end

function fixY!(layer::L, x::Matrix) where {L <: Union{BPLayer}}
    @extract layer: K N M 
    @extract layer: x̂ x̂cav Δ m mcav σ 
    @assert size(x) == size(x̂)
    x̂ .= x
    x̂cav .= reshape(x, 1, N, M)
    Δ .= 0
end

function getW(layer::L) where L <: Union{BPLayer}
    return sign.(layer.m)
end

function forward(layer::L, x) where L <: Union{BPLayer}
    @extract layer: N K
    @assert size(x, 1) == N
    W = getW(layer)
    return sign.(W*x .+ 1e-10)
end

