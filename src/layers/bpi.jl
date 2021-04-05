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

    type::Symbol
    top_layer::AbstractLayer
    bottom_layer::AbstractLayer
    weight_mask
    isfrozen::Bool
end

function BPILayer(K::Int, N::Int, M::Int; 
            density=1., isfrozen=false, type=:bpi)
    # for variables W
    x̂ = zeros(F, N, M)
    Δ = zeros(F, N, M)
    
    m = zeros(F, K, N)
    σ = zeros(F, K, N)
    
    Bup = zeros(F, K, M)
    B = zeros(F, N, M)
    A = zeros(F, N, M)
    
    H = zeros(F, K, N)
    Hext = zeros(F, K, N)
    
    ω = zeros(F, K, M)
    V = zeros(F, K, M)
    
    weight_mask = rand(F, K, N) .< density

    return BPILayer(-1, K, N, M,
            x̂, Δ, m, σ,
            Bup, B, A, 
            H, Hext,
            ω, V,
            type,
            DummyLayer(), DummyLayer(),
            weight_mask, isfrozen)
end

function update!(layer::BPILayer, reinfpar; mode=:both)
    @extract layer: K N M weight_mask
    @extract layer: x̂ Δ m  σ 
    @extract layer: Bup B A H Hext ω  V
    @extract layer: bottom_layer top_layer
    @extract reinfpar: r y
    Δm = 0.

    if mode == :forw || mode == :both
        if !isbottomlayer(layer)
            @tullio x̂[i,a] = tanh(bottom_layer.Bup[i,a] + B[i,a])
            Δ .= 1 .- x̂.^2
        end
        
        @tullio ω[k,a] = m[k,i] * x̂[i,a]
        V .= .√(σ * x̂.^2 + m.^2 * Δ + σ * Δ .+ 1f-8)
        @tullio Bup[k,a] = atanh2Hm1(-ω[k,a] / V[k,a]) avx=false

        @assert all(isfinite, Bup)
    end
    if mode == :back || mode == :both
        Btop = top_layer.B 
        @assert size(Btop) == (K, M)
        @tullio g[k,a] := compute_g(Btop[k,a], ω[k,a], V[k,a])  avx=false
        if layer.type == :bpi
            @tullio gcav[k,i,a] := compute_g(Btop[k,a], ω[k,a]- m[k,i] * x̂[i,a], V[k,a])  avx=false
        end
        # @tullio Γ[k,a] := compute_Γ(Btop[k,a], ω[k,a], V[k,a])
        
        if !isbottomlayer(layer)
            if layer.type == :bpi
                @tullio B[i,a] = m[k,i] * gcav[k,i,a]
            # if layer.type == :bpi2
            #     @tullio B[i,a] = m[k,i] * (g[k,i,a]
            else
                # A .= (m.^2 + σ)' * Γ - σ' * g.^2
                @tullio B[i,a] = m[k,i] * g[k,a]
            end
        end

        if !isfrozen(layer)
            if layer.type == :bpi 
                @tullio Hin[k,i] := gcav[k,i,a] * x̂[i,a]
            else
                @tullio Hin[k,i] := g[k,a] * x̂[i,a]
            end
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
            mnew = tanh.(H) .* weight_mask
            Δm = maximum(abs, m .- mnew)
            m .= mnew
            σ .= (1 .- m.^2) .* weight_mask
            @assert all(isfinite, m)
        end
    end
    
    return Δm
end

function initrand!(layer::L) where {L <: Union{BPILayer}}
    @extract layer: K N M weight_mask
    @extract layer: x̂ Δ m σ 
    @extract layer: B A ω H V Hext
    # TODO reset all variables
    ϵ = 1f-1
    H .= ϵ .* randn!(similar(m)) + Hext
    m .= tanh.(H) .* weight_mask
    σ .= (1 .- m.^2) .* weight_mask
end

function fixY!(layer::L, x::AbstractMatrix) where {L <: Union{BPILayer}}
    @extract layer: K N M 
    @extract layer: x̂ Δ m σ 
    @assert size(x) == size(x̂)
    x̂ .= x
    Δ .= 0
end

function getW(layer::BPILayer)
    return sign.(layer.m) .* layer.weight_mask
end

function forward(layer::BPILayer, x)
    @extract layer: N K
    @assert size(x, 1) == N
    W = getW(layer)
    return sign.(W*x .+ 1f-10)
end

function fixW!(layer::BPILayer, w=1.)
    @extract layer: K N M m σ weight_mask
    m .= w .* weight_mask
    σ .= 0
end

