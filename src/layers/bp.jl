
mutable struct BPLayer{A2,A3,M} <: AbstractLayer
    l::Int
    K::Int
    N::Int
    M::Int

    x̂::A2
    x̂cav::A3 
    Δ::A2

    m::A2
    mcav::A3 
    σ::A2

    Bup::A2
    B::A2 
    Bcav::A3 
    A::A2

    H::A2
    Hext::A2
    Hcav::A3

    ω::A2
    ωcav::A3 
    V::A2

    top_layer::AbstractLayer
    bottom_layer::AbstractLayer

    weight_mask::M
    isfrozen::Bool
end

@functor BPLayer


function BPLayer(K::Int, N::Int, M::Int; 
        density=1., isfrozen=false)
    x̂ = zeros(F, N, M)
    x̂cav = zeros(F, K, N, M)
    Δ = zeros(F, N, M)
    
    m = zeros(F, K, N)
    mcav = zeros(F, K, N, M)
    σ = zeros(F, K, N)
    
    Bup = zeros(F, K, M)
    B = zeros(F, N, M)
    Bcav = zeros(F, K, N, M)
    A = zeros(F, N, M)
    
    H = zeros(F, K, N)
    Hext = zeros(F, K, N)
    Hcav = zeros(F, K, N, M)
    
    ω = zeros(F, K, M)
    ωcav = zeros(F, K, N, M)
    V = zeros(F, K, M)
    
    weight_mask = rand(K, N) .< density

    return BPLayer(-1, K, N, M,
            x̂, x̂cav, Δ, m, mcav, σ,
            Bup, B, Bcav, A, 
            H, Hext, Hcav,
            ω, ωcav, V,
            DummyLayer(), DummyLayer(),
            weight_mask, isfrozen)
end

@gpu function compute_g(B, ω, V)
    1/√V * GH2(B, -ω / √V) 
end

function update!(layer::BPLayer, reinfpar; mode=:both)
    @extract layer: K N M weight_mask
    @extract layer: x̂ x̂cav Δ m mcav σ 
    @extract layer: Bup B Bcav A H Hext Hcav ω ωcav V
    @extract layer: bottom_layer top_layer
    @extract reinfpar: r y
    Δm = 0.

    if mode == :forw || mode == :both
        if !isbottomlayer(layer)
            @tullio x̂cav[k,i,a] = tanh(bottom_layer.Bup[i,a] + Bcav[k,i,a])
            @tullio x̂[i,a] = tanh(bottom_layer.Bup[i,a] + B[i,a])
            Δ .= 1 .- x̂.^2
        end
        
        @tullio ω[k,a] = mcav[k,i,a] * x̂cav[k,i,a]
        @tullio ωcav[k,i,a] = ω[k,a] - mcav[k,i,a] * x̂cav[k,i,a]
        V .= σ * x̂.^2 + m.^2 * Δ + σ * Δ .+ 1f-8
        @tullio Bup[k,a] = atanh2Hm1(-ω[k,a] / √V[k,a]) avx=false
    end

    if mode == :back || mode == :both
        ## BACKWARD 
        Btop = top_layer.B 
        @assert size(Btop) == (K, M)
        gcav = compute_g.(reshape(Btop,K,1,M), ωcav, reshape(V,K,1,M))
        g = compute_g.(Btop, ω, V)
        @tullio gcav[k,i,a] := compute_g(Btop[k,a], ωcav[k,i,a], V[k,a])  avx=false
        @tullio g[k,a] := compute_g(Btop[k,a], ω[k,a], V[k,a])  avx=false
        # @tullio Γ[k,a] := compute_Γ(Btop[k,a], ω[k,a], V[k,a])
        
        if !isbottomlayer(layer)
            # A .= (m.^2 + σ)' * Γ - σ' * g.^2
            @tullio B[i,a] = mcav[k,i,a] * gcav[k,i,a]
            @tullio Bcav[k,i,a] = B[i,a] - mcav[k,i,a] * gcav[k,i,a]
        end

        if !isfrozen(layer)
            @tullio Hin[k,i] := gcav[k,i,a] * x̂cav[k,i,a] 
            if y > 0 # focusing
                tγ = tanh(r)
                @tullio mjs[k,i] := tanh(Hin[k,i])
                @tullio mfoc[k,i] := tanh((y-1)*atanh(mjs[k,i]*tγ)) * tγ
                @tullio Hfoc[k,i] := atanh(mfoc[k,i])
                @tullio H[k,i] = Hin[k,i] + Hfoc[k,i] + Hext[k,i]
            else
                # reinforcement
                @tullio H[k,i] = Hin[k,i] + r*H[k,i] + Hext[k,i]
            end
            @tullio Hcav[k,i,a] = H[k,i] - gcav[k,i,a] * x̂cav[k,i,a]
            @tullio mcav[k,i,a] = tanh(Hcav[k,i,a]) * weight_mask[k,i]
            mnew = tanh.(H) .* weight_mask
            Δm = maximum(abs, m .- mnew) 
            m .= mnew
            σ .= (1 .- m.^2) .* weight_mask    
            @assert all(isfinite, m)
        end
        
    end
    
    return Δm
end

function initrand!(layer::L) where {L <: Union{BPLayer}}
    @extract layer: K N M weight_mask
    @extract layer: x̂ x̂cav Δ m mcav σ Hext
    @extract layer: B Bcav A ω H Hcav ωcav V
    # TODO reset all variables
    ϵ = 1f-1
    H .= ϵ .* randn!(similar(Hext)) + Hext
    m .= tanh.(H) .* weight_mask
    mcav .= m .* weight_mask 
    σ .= (1 .- m.^2) .* weight_mask
end

function fixY!(layer::L, x::AbstractMatrix) where {L <: Union{BPLayer}}
    @extract layer: K N M 
    @extract layer: x̂ x̂cav Δ m mcav σ 
    @assert size(x) == size(x̂)
    x̂ .= x
    x̂cav .= reshape(x, 1, N, M)
    Δ .= 0
end

function getW(layer::L) where L <: Union{BPLayer}
    return sign.(layer.m) .* layer.weight_mask
end

function forward(layer::L, x) where L <: Union{BPLayer}
    @extract layer: N K
    @assert size(x, 1) == N
    W = getW(layer)
    return sign.(W*x .+ 1f-10)
end

function fixW!(layer::L, w=1.) where {L <: Union{BPLayer}}
    @extract layer: K N M m σ mcav weight_mask
    m .= w .* weight_mask
    mcav .= m .* weight_mask
    σ .= 0
end
