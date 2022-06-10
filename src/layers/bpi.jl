mutable struct BPILayer{A2<:AbstractMatrix, M, ACT<:AbstractChannel} <: AbstractLayer
    l::Int
    K::Int
    N::Int
    M::Int
    ϵinit

    x̂::A2
    Δ::A2

    m::A2
    σ::A2

    B::A2
    A::A2 
    
    H::A2
    Hext::A2
    
    ω::A2
    V::A2

    top_layer::AbstractLayer
    bottom_layer::AbstractLayer
    

    weight_mask::M
    isfrozen::Bool

    act::ACT
end

@functor BPILayer

function BPILayer(K::Int, N::Int, M::Int, ϵinit; 
            density=1., isfrozen=false, act=nothing)
    x̂ = zeros(F, N, M)
    Δ = zeros(F, N, M)
    
    m = zeros(F, K, N)
    σ = zeros(F, K, N)
    
    B = zeros(F, N, M)
    A = zeros(F, N, M)
    
    H = zeros(F, K, N)
    Hext = zeros(F, K, N)
    
    ω = zeros(F, K, M)
    V = zeros(F, K, M)
    
    weight_mask = rand(K, N) .< density

    if act === nothing
        act = :sign
    end
    act = channel(act)

    return BPILayer(-1, K, N, M, ϵinit,
            x̂, Δ, 
            m, σ,
            B, A, 
            H, Hext,
            ω, V,
            DummyLayer(), DummyLayer(),
            weight_mask, isfrozen,
            act)
end

function update!(layer::BPILayer, reinfpar; mode=:both)
    @extract layer: K N M weight_mask
    @extract layer: x̂ Δ m  σ 
    @extract layer: Bup B A H Hext ω  V
    @extract layer: bottom_layer top_layer act
    @extract reinfpar: r y ψ l

    Δm = 0.
    rl = r[l]
    @assert y == 0 # deprecate focusing


    if mode == :forw || mode == :both
        if !isbottomlayer(layer)
            compute_x̂!(x̂, B, A, bottom_layer)
            compute_x̂cav!(x̂cav, Bcav, A, bottom_layer)
            compute_Δ!(Δ, B, A, bottom_layer)
        end
        
        @tullio ω[k,a] = m[k,i] * x̂[i,a]
        V .=  σ * x̂.^2 + m.^2 * Δ + σ * Δ .+ 1f-8
    end
    if mode == :back || mode == :both
        Btop = top_layer.B 
        @assert size(Btop) == (K, M)
        @tullio g[k,a] := compute_g(Btop[k,a], ω[k,a], V[k,a])  avx=false
        @tullio gcav[k,i,a] := compute_g(Btop[k,a], ω[k,a]- m[k,i] * x̂[i,a], V[k,a])  avx=false
        # @tullio Γ[k,a] := compute_Γ(Btop[k,a], ω[k,a], V[k,a])
        
        if !isbottomlayer(layer)
            @tullio B[i,a] = m[k,i] * gcav[k,i,a]
        end

        if !isfrozen(layer)
            @tullio Hin[k,i] := gcav[k,i,a] * x̂[i,a]
            
            # reinforcement 
            @tullio Hnew[k,i] := Hin[k,i] + rl * H[k,i] + Hext[k,i]
            
            # H .= ψ[l] .* H .+ (1-ψ[l]) .* Hnew
            H .= Hnew

            mnew = (ψ[l] .* m .+ (1-ψ[l]) .* tanh.(H)) .* weight_mask
            # mnew = tanh.(H) .* weight_mask
            Δm = mean(abs.(m .- mnew))
            m .= mnew
            σ .= (1 .- m.^2) .* weight_mask
            # @assert all(isfinite, m)
        end
    end
    
    return Δm
end

function initrand!(layer::L) where {L <: Union{BPILayer}}
    @extract layer: K N M weight_mask ϵinit
    @extract layer: x̂ Δ m σ 
    @extract layer: B A ω H V Hext
    # TODO reset all variables
    H .= ϵinit .* randn!(similar(m)) + Hext
    m .= tanh.(H) .* weight_mask
    σ .= (1 .- m.^2) .* weight_mask
end

function fix_input!(layer::L, x::AbstractMatrix) where {L <: Union{BPILayer}}
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

