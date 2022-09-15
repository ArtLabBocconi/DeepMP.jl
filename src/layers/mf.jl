###########################
#       MeanField LAYER
#######################################
mutable struct MeanFieldLayer <: AbstractLayer
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

@functor MeanFieldLayer

function MeanFieldLayer(K::Int, N::Int, M::Int, ϵinit; density=1, isfrozen=false)
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
    
    return MeanFieldLayer(-1, K, N, M, ϵinit,
            x̂, Δ, m, σ,
            Bup, B, A, 
            H, Hext,
            g, ω, V,
            DummyLayer(), DummyLayer(),
            weight_mask, isfrozen)
end


function update!(layer::MeanFieldLayer, reinfpar; mode=:both)
    @extract layer: K N M weight_mask
    @extract layer: x̂ Δ m σ 
    @extract layer: Bup B  A H Hext ω  V g
    @extract layer: bottom_layer top_layer
    @extract reinfpar: r ψ l
    
    Δm = 0.
    rl = r[l]

    if mode == :forw || mode == :both
        ## FORWARD
        if !isbottomlayer(layer)
            bottBup = bottom_layer.Bup
            @tullio x̂[i,a] = tanh(bottBup[i,a] + B[i,a])
            Δ .= 1 .- x̂.^2
        end
        
        V .= σ * x̂.^2 + m.^2 * Δ 
        @tullio ω[k,a] = m[k,i] * x̂[i,a]
        V .+= σ * Δ .+ 1f-8 
        @tullio Bup[k,a] = atanh2Hm1(-ω[k,a] / √V[k,a]) avx=false
    end

    if mode == :back || mode == :both
        ## BACKWARD 
        Btop = top_layer.B 
        @assert size(Btop) == (K, M)
        @tullio g[k,a] = compute_g(Btop[k,a], ω[k,a], √V[k,a])  avx=false
        
        if !isbottomlayer(layer)
            @tullio B[i,a] = m[k,i] * g[k,a]
        end

        if !isfrozen(layer)
            @tullio Hin[k,i] := g[k,a] * x̂[i,a]
            @tullio Hnew[k,i] := Hin[k,i]  + rl * H[k,i] + Hext[k,i]
            H .= Hnew
            # H .= ψ[l] .* H .+ (1-ψ[l]) .* Hnew
            
            mnew = ψ[l] .* m .+ (1-ψ[l]) .* tanh.(H) .* weight_mask
            # mnew = tanh.(H) .* weight_mask
            Δm = mean(abs.(m .- mnew))
            m .= mnew
            σ .= (1 .- m.^2) .* weight_mask
        end
    end
    
    return Δm
end


function initrand!(layer::MeanFieldLayer)
    @extract layer: K N M weight_mask ϵinit
    @extract layer: x̂  Δ m σ 
    @extract layer: B A ω H  V Hext
    H .= ϵinit .* randn!(similar(m)) + Hext
    m .= tanh.(H) .* weight_mask
    σ .= (1 .- m.^2) .* weight_mask
end

function fix_input!(layer::L, x::AbstractMatrix) where {L <: Union{MeanFieldLayer}}
    @extract layer: K N M 
    @extract layer: x̂ Δ m  σ 
    @assert size(x) == size(x̂)
    x̂ .= x
    Δ .= 0
end

function getW(layer::L) where L <: Union{MeanFieldLayer}
    return sign.(layer.m) .* layer.weight_mask
end

function forward(layer::L, x) where L <: Union{MeanFieldLayer}
    @extract layer: N K
    @assert size(x, 1) == N
    W = getW(layer)
    return sign.(W*x .+ 1f-10)
end

function fixW!(layer::L, w=1.) where {L <: Union{MeanFieldLayer}}
    @extract layer: K N M m σ  weight_mask
    m .= w .* weight_mask
    σ .= 0
end
