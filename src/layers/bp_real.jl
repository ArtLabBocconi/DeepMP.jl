
mutable struct BPRealLayer{A2,A3,M} <: AbstractLayer
    l::Int
    K::Int
    N::Int
    M::Int
    ϵinit

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
    Ω::A2
    Ωext::A2

    ω::A2
    ωcav::A3 
    V::A2

    top_layer::AbstractLayer
    bottom_layer::AbstractLayer

    weight_mask::M
    isfrozen::Bool
end

@functor BPRealLayer


function BPRealLayer(K::Int, N::Int, M::Int, ϵinit; 
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
    Ω = zeros(F, K, N)
    Ωext = zeros(F, K, N)

    
    ω = zeros(F, K, M)
    ωcav = zeros(F, K, N, M)
    V = zeros(F, K, M)
    
    weight_mask = rand(K, N) .< density

    return BPRealLayer(-1, K, N, M, ϵinit,
            x̂, x̂cav, Δ, m, mcav, σ,
            Bup, B, Bcav, A, 
            H, Hext, Hcav, Ω, Ωext,
            ω, ωcav, V,
            DummyLayer(), DummyLayer(),
            weight_mask, isfrozen)
end

function update!(layer::BPRealLayer, reinfpar; mode=:both)
    @extract layer: K N M weight_mask
    @extract layer: x̂ x̂cav Δ m mcav σ 
    @extract layer: Bup B Bcav A H Hext Hcav Ω Ωext ω ωcav V
    @extract layer: bottom_layer top_layer
    @extract reinfpar: r y ψ
    Δm = 0.

    if mode == :forw || mode == :both
        if !isbottomlayer(layer)
            bottBup = bottom_layer.Bup # issue https://github.com/mcabbott/Tullio.jl/issues/96
            @tullio x̂cav[k,i,a] = tanh(bottBup[i,a] + Bcav[k,i,a])
            @tullio x̂[i,a] = tanh(bottBup[i,a] + B[i,a])
            # @tullio x̂cavnew[k,i,a] := tanh(bottBup[i,a] + Bcav[k,i,a])
            # @tullio x̂new[i,a] := tanh(bottBup[i,a] + B[i,a])
            # x̂ .= ψ .* x̂ .+ (1-ψ) .* x̂new
            # x̂cav .= ψ .* x̂cav .+ (1-ψ) .* x̂cavnew   
            Δ .= 1 .- x̂.^2
        end
        # @assert all(isfinite, x̂)
        # @assert all(isfinite, x̂cav)

        
        @tullio ω[k,a] = mcav[k,i,a] * x̂cav[k,i,a]
        @tullio ωcav[k,i,a] = ω[k,a] - mcav[k,i,a] * x̂cav[k,i,a]
        V .= .√(σ * x̂.^2 + m.^2 * Δ + σ * Δ .+ 1f-8)
        @tullio Bup[k,a] = atanh2Hm1(-ω[k,a] / V[k,a]) avx=false
        # @assert all(isfinite, ω)
        # @assert all(isfinite, V)
        # @assert all(isfinite, ωcav)
        # @assert all(isfinite, Bup)
    end

    if mode == :back || mode == :both
        ## BACKWARD 
        Btop = top_layer.B 
        @assert size(Btop) == (K, M)
        # @assert all(isfinite, Btop)

        @tullio gcav[k,i,a] := compute_g(Btop[k,a], ωcav[k,i,a], V[k,a])  avx=false
        @tullio g[k,a] := compute_g(Btop[k,a], ω[k,a], V[k,a])  avx=false
        # @tullio Γ[k,a] := compute_Γ(Btop[k,a], ω[k,a], V[k,a])
        grad_g(B, ω, V)::F = gradient(ω -> compute_g(B, ω, V)::F, ω)[1]
        Γ = grad_g.(B, ω, V)

        # @assert all(isfinite, g)
        # @assert all(isfinite, gcav)
        
        if !isbottomlayer(layer)
            # A .= (m.^2 + σ)' * Γ - σ' * g.^2
            @tullio B[i,a] = mcav[k,i,a] * gcav[k,i,a]
            @tullio Bcav[k,i,a] = B[i,a] - mcav[k,i,a] * gcav[k,i,a]
            # @assert all(isfinite, B)
            # @assert all(isfinite, Bcav)
        end

        if !isfrozen(layer)
            @tullio Hin[k,i] := gcav[k,i,a] * x̂cav[k,i,a] 
            @tullio Ωin[k,i] := (x̂[i,a]^2 + Δ[i,a]) * Γ[k,a] - Δ[i,a] * g[k,a]^2
            if y > 0 # focusing
                @assert false "focusing not fully supported"
                tγ = tanh(r)
                @tullio mjs[k,i] := tanh(Hin[k,i])
                @tullio mfoc[k,i] := tanh((y-1)*atanh(mjs[k,i]*tγ)) * tγ
                @tullio Hfoc[k,i] := atanh(mfoc[k,i])
                @tullio Hnew[k,i] := Hin[k,i] + Hfoc[k,i] + Hext[k,i]
            else
                # reinforcement
                @tullio Hnew[k,i] := Hin[k,i] + r*H[k,i] + Hext[k,i]
                @tullio Ωnew[k,i] := Ωin[k,i] + r*Ω[k,i] + Ωext[k,i]
            end
            @tullio Hcavnew[k,i,a] := Hnew[k,i] - gcav[k,i,a] * x̂cav[k,i,a]
            H .= ψ .* H .+ (1 - ψ) .* Hnew 
            Ω .= ψ .* Ω .+ (1 - ψ) .* Ωnew 
            Hcav .= ψ .* Hcav .+ (1 - ψ) .* Hcavnew 
            # H .= Hnew 
            # Hcav .= Hcavnew 
            
            @tullio mcavnew[k,i,a] := (Hcav[k,i,a] / Ω[k,i]) * weight_mask[k,i]
            # mcav .= ψ .* mcav .+ (1 - ψ) .* mcavnew
            mcav .= mcavnew
            
            # @assert all(isfinite, H)
            # @assert all(isfinite, Hcav)


            mnew = (H ./ Ω) .* weight_mask
            Δm = mean(abs.(m .- mnew))
            m .= mnew
            σ .= (1 ./ Ω) .* weight_mask    
            @assert all(isfinite, m)
            @assert all(isfinite, σ)
        end
        
    end
    
    return Δm
end

function initrand!(layer::L) where {L <: Union{BPRealLayer}}
    @extract layer: K N M weight_mask ϵinit
    @extract layer: x̂ x̂cav Δ m mcav σ 
    @extract layer: B Bcav A ω H Hcav Hext Ω Ωext ωcav V
    # TODO reset all variables
    H .= ϵinit .* randn!(similar(Hext)) + Hext
    Ω .= 1 .+ Ωext
    
    m .= H ./ Ω .* weight_mask
    mcav .= m .* weight_mask 
    σ .= 1 ./ Ω .* weight_mask
end

function fix_input!(layer::L, x::AbstractMatrix) where {L <: Union{BPRealLayer}}
    @extract layer: K N M 
    @extract layer: x̂ x̂cav Δ m mcav σ 
    @assert size(x) == size(x̂)
    x̂ .= x
    x̂cav .= reshape(x, 1, N, M)
    Δ .= 0
end

function getW(layer::L) where L <: Union{BPRealLayer}
    return layer.m .* layer.weight_mask
end

function forward(layer::L, x) where L <: Union{BPRealLayer}
    @extract layer: N K
    @assert size(x, 1) == N
    W = getW(layer)
    return sign.(W*x .+ 1f-10)
end

function fixW!(layer::L, w=1.) where {L <: Union{BPRealLayer}}
    @extract layer: K N M m σ mcav weight_mask
    m .= w .* weight_mask
    mcav .= m .* weight_mask
    σ .= 0
end
