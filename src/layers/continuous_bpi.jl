"""
BPI with real weights
"""
mutable struct CBPILayer <: AbstractLayer
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
    Ω
    Ωext
    ω 
    V

    type::Symbol
    top_layer::AbstractLayer
    bottom_layer::AbstractLayer
    weight_mask
    isfrozen::Bool
end

@functor CBPILayer

function CBPILayer(K::Int, N::Int, M::Int, ϵinit; 
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
    Ω = zeros(F, K, N)
    Ωext = zeros(F, K, N)
    
    ω = zeros(F, K, M)
    V = zeros(F, K, M)
    
    weight_mask = rand(F, K, N) .< density

    return CBPILayer(-1, K, N, M, ϵinit,
            x̂, Δ, m, σ,
            Bup, B, A, 
            H, Hext,
            Ω, Ωext,
            ω, V,
            type,
            DummyLayer(), DummyLayer(),
            weight_mask, isfrozen)
end

function update!(layer::CBPILayer, reinfpar; mode=:both)
    @extract layer: K N M weight_mask
    @extract layer: x̂ Δ m  σ 
    @extract layer: Bup B A H Hext ω  V  Ω  Ωext
    @extract layer: bottom_layer top_layer
    @extract reinfpar: r y ψ l
    
    Δm = 0.
    rl = r[l]

    if mode == :forw || mode == :both
        if !isbottomlayer(layer)
            bottBup = bottom_layer.Bup
            @tullio x̂[i,a] = tanh(bottBup[i,a] + B[i,a])
            Δ .= 1 .- x̂.^2
        end
        
        @tullio ω[k,a] = m[k,i] * x̂[i,a]
        V .= .√(σ * x̂.^2 + m.^2 * Δ + σ * Δ .+ 1f-8)
        @tullio Bup[k,a] = atanh2Hm1(-ω[k,a] / V[k,a]) avx=false
    end
    if mode == :back || mode == :both
        Btop = top_layer.B 
        @assert size(Btop) == (K, M)
        @tullio g[k,a] := compute_g(Btop[k,a], ω[k,a], V[k,a])  avx=false
        @tullio gcav[k,i,a] := compute_g(Btop[k,a], ω[k,a] - m[k,i] * x̂[i,a], V[k,a])  avx=false
        
        # @tullio Γ[k,a] := compute_Γ(Btop[k,a], ω[k,a], V[k,a])
        grad_g(B, ω, V) = ForwardDiff.derivative(ω -> compute_g(B, ω, V), ω)
        Γ = -grad_g.(Btop, ω, V)
        # @show count(!isfinite, Γ)
        if !all(isfinite, Γ)
            # @warn mean((!isfinite).(Γ))
            # Γa = Array(Γ)
            # iinf = (!isfinite).(Γa)
            # @show Γa[iinf] Btop[iinf] ω[iinf] V[iinf]
            # Γb = grad_g.(Array{Float64}(Btop), Array{Float64}(ω), Array{Float64}(V))
            # @show Γb[iinf] Array(Btop)[iinf] Array(ω)[iinf] Array(V)[iinf]
            Γ[(!isfinite).(Γ)] .= 0
        end
        @assert all(isfinite, g)
        @assert all(isfinite, gcav)
        @assert all(isfinite, Γ)
        
        if !isbottomlayer(layer)
            @tullio B[i,a] = m[k,i] * gcav[k,i,a]
            #TODO add A
        end

        if !isfrozen(layer)
            @tullio Hin[k,i] := gcav[k,i,a] * x̂[i,a]
            @tullio Ωin[k,i] := (x̂[i,a]^2 + Δ[i,a]) * Γ[k,a] - Δ[i,a] * g[k,a]^2
            if y > 0 # focusing
                @assert false
            else
                # reinforcement 
                @tullio Ωnew[k,i] := Ωin[k,i] + rl*Ω[k,i] + Ωext[k,i]
                @tullio Hnew[k,i] := Hin[k,i] + rl*H[k,i] + Hext[k,i]
            end
            # H .= ψ[l] .* H .+ (1-ψ[l]) .* Hnew
            H .= Hnew
            # @show mean(Ωnew .> 0)
            # @show mean(Ωnew) mean(abs.(H))
            Ω .= max.(1f-6, Ωnew)

            mnew = ψ[l] .* m .+ (1-ψ[l]) .* (H ./ Ω) .* weight_mask
            Δm = mean(abs.(m .- mnew))
            m .= mnew
            # σ .= ψ[l] .* σ .+ (1-ψ[l]) .* (1 ./ Ω) .* weight_mask
            σ .= (1 ./ Ω) .* weight_mask
            @assert all(σ .>= 0)
        end
    end
    
    return Δm
end

function initrand!(layer::L) where {L <: Union{CBPILayer}}
    @extract layer: K N M weight_mask ϵinit
    @extract layer: x̂ Δ m σ 
    @extract layer: B A ω H Ω V Hext Ωext
    μ = ϵinit / √N
    stdev = ϵinit / √N
    # stdev = 0.1
  
    H .= (μ / stdev^2) .* randn!(similar(m)) .+ Hext
    Ω .= (1 / stdev^2) .* fill!(similar(m), 1) .+ Ωext
    m .= (H ./ Ω) .* weight_mask
    σ .= (1 ./ Ω) .* weight_mask
end

function fix_input!(layer::L, x::AbstractMatrix) where {L <: Union{CBPILayer}}
    @extract layer: K N M 
    @extract layer: x̂ Δ m σ 
    @assert size(x) == size(x̂)
    x̂ .= x
    Δ .= 0
end

function getW(layer::CBPILayer)
    return layer.m .* layer.weight_mask
end

function forward(layer::CBPILayer, x)
    @extract layer: N K
    @assert size(x, 1) == N
    W = getW(layer)
    return sign.(W*x .+ 1f-10)
end

function fixW!(layer::CBPILayer, w=1.)
    @extract layer: K N M m σ weight_mask
    m .= w .* weight_mask
    σ .= 0
end

