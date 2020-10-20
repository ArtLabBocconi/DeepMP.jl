
mutable struct BPLayer2 <: AbstractLayer
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

    B 
    Bcav 
    A 
    
    H
    Hcav 

    ω 
    ωcav 
    V

    top_layer::AbstractLayer
    bottom_layer::AbstractLayer

    allhext

    weight_mask
end


function BPLayer2(K::Int, N::Int, M::Int; density=1.)
    # for variables W
    x̂ = zeros(N, M)
    x̂cav = zeros(K, N, M)
    Δ = zeros(N, M)
    
    m = zeros(K, N)
    mcav = zeros(K, N, M)
    σ = zeros(K, N)
    
    B = zeros(N, M)
    Bcav = zeros(K, N, M)
    A = zeros(N, M)
    
    H = zeros(K, N)
    Hcav = zeros(K, N, M)
    
    ω = zeros(K, M)
    ωcav = zeros(K, N, M)
    V = zeros(K, M)
    
    allhext = [zeros(N) for i=1:K]

    weight_mask = [[rand() < density ? 1 : 0 for i=1:N] for i=1:K]

    return BPLayer2(-1, K, N, M,
            x̂, x̂cav, Δ, m, mcav, σ,
            B, Bcav, A, H, Hcav,
            ω, ωcav, V,
            DummyLayer(), DummyLayer(),
            allhext, weight_mask)
end


# function initYBottom!(layer::L, a::Int) where {L <: Union{BPLayer2}}

#     @assert isbottomlayer(layer)

#     my = allmy[a]
#     x = layer.bottom_layer.x
#     for i=1:N
#         my[i] = x[i, a]
#         mycav = allmycav[a]
#         for k=1:K
#             mycav[k][i] = x[i, a]
#         end
#     end
# end

function get_AB(layer::AbstractLayer)
    B = vcat([b' for b in layer.allpd]...)
    return 0, B 
end

function compute_g(B, ω, V)
    1/√V * GH(B, -ω / √V)
end

function update!(layer::L, reinfpar) where {L <: Union{BPLayer2}}
    @extract layer: K N M
    @extract layer: x̂ x̂cav Δ m mcav σ 
    @extract layer: B Bcav A H Hcav ω ωcav V
    @extract layer: bottom_layer top_layer
    @extract reinfpar: r
    ## FORWARD
    # ωbottom, Vbottom = get_ωV(layer.bottom_layer)
    # @tullio x̂cav[k,i,a] = compute_x(Bcav[k,i,a], ωbottom[i,a], Vbottom[i,a])
    # @tullio x̂[k,a] = compute_x(B[i,a], ωbottom[i,a], Vbottom[i,a])
    # Δ .= 1 .- x̂.^2 
    
    @tullio ω[k,a] = mcav[k,i,a] * x̂cav[k,i,a]
    @tullio ωcav[k,i,a] = ω[k,a] - mcav[k,i,a] * x̂cav[k,i,a]
    V .= σ * x̂.^2 + m.^2 * Δ + σ * Δ

    # # BACKWARD 
    Atop, Btop = get_AB(top_layer)
    @tullio gcav[k,i,a] := compute_g(Btop[k,a], ωcav[k,i,a], V[k,a])  avx=false
    @tullio g[k,a] := compute_g(Btop[k,a], ω[k,a], V[k,a])  avx=false
    # @tullio Γ[k,a] := compute_Γ(Btop[k,a], ω[k,a], V[k,a])
    
    # A .= (m.^2 + σ)' * Γ - σ' * g.^2
    @tullio B[i,a] = mcav[k,i,a] * gcav[k,i,a]
    @tullio Bcav[k,i,a] = B[i,a] - mcav[k,i,a] * gcav[k,i,a]
    
    
    @tullio H[k,i] = gcav[k,i,a] * x̂cav[k,i,a] + r * H[k,i]  
    @tullio Hcav[k,i,a] = H[k,i] - gcav[k,i,a] * x̂cav[k,i,a]
    mcav .= tanh.(Hcav)
    mnew = tanh.(H)
    Δm = maximum(abs, m .- mnew) 
    m .= mnew
    σ .= 1 .- m.^2    
    
    return Δm
end


function initrand!(layer::L) where {L <: Union{BPLayer2}}
    @extract layer: K N M
    @extract layer: x̂ x̂cav Δ m mcav σ 
    @extract layer: B Bcav A ω H Hcav ωcav V
    # TODO reset all variables
    ϵ = 1e-0
    H .= ϵ .* randn(K, N)
    m .= tanh.(H)
    mcav .= m
    σ .= 1 .- m.^2
end

function fixY!(layer::L, x::Matrix) where {L <: Union{BPLayer2}}
    @extract layer: K N M 
    @extract layer: x̂ x̂cav Δ m mcav σ 
    @assert size(x) == size(x̂)
    x̂ .= x
    x̂cav .= reshape(x, 1, N, M)
    Δ .= 0
end

function getW(layer::L) where L <: Union{BPLayer2}
    return sign.(layer.m)
end

function forward(layer::L, x) where L <: Union{BPLayer2}
    @extract layer: N K
    @assert size(x, 1) == N
    W = getW(layer)
    return sign.(W*x .+ 1e-10)
end

