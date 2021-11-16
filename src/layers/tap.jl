###########################
#       TAP LAYER
#######################################
mutable struct TapLayer <: AbstractLayer
    l::Int

    K::Int
    N::Int
    M::Int
    ϵinit


    x̂old
    x̂
    Δ

    mold 
    m 
    σ 

    Bup
    B 
    A 
    
    H   # = Hcav + Hext
    Hext
    
    
    
    gold 
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
    x̂old = zeros(F, N, M)
    x̂ = zeros(F, N, M)
    Δ = zeros(F, N, M)
    
    mold = zeros(F, K, N)
    m = zeros(F, K, N)
    σ = zeros(F, K, N)
    
    Bup = zeros(F, K, M)
    B = zeros(F, N, M)
    A = zeros(F, N, M)
    
    H = zeros(F, K, N)
    Hext = zeros(F, K, N)
    
    gold = zeros(F, K, M)
    g = zeros(F, K, M)
    ω = zeros(F, K, M)
    V = zeros(F, K, M)
    
    weight_mask = rand(K, N) .< density
    
    return TapLayer(-1, K, N, M, ϵinit,
            x̂old, x̂, Δ, mold, m, σ,
            Bup, B, A, 
            H, Hext,
            gold, g, ω, V,
            DummyLayer(), DummyLayer(),
            weight_mask, isfrozen)
end


function update!(layer::TapLayer, reinfpar; mode=:both)
    @extract layer: K N M weight_mask
    @extract layer: x̂ Δ m σ x̂old mold
    @extract layer: Bup B A H Hext ω  V g gold
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

        σ .= 1 .- m.^2
        
        V .= m.^2 * Δ + σ * x̂.^2 + σ * Δ .+ 1f-8

        @tullio ω[k,a] = m[k,i] * x̂[i,a]
        # Terms below are 0 for first iteration
        @tullio ω[k,a] += - gold[k,a] * σ[k,i] * x̂[i,a] * x̂old[i,a]
        @tullio ω[k,a] += - gold[k,a] * m[k,i] * mold[k,i] * Δ[i,a]
        @tullio ω[k,a] += + gold[k,a]^2 * σ[k,i] * mold[k,i] * x̂old[i,a] * Δ[i,a]
        
        @tullio Bup[k,a] = atanh2Hm1(-ω[k,a] / √V[k,a]) avx=false
    end

    if mode == :back || mode == :both

        ## BACKWARD 
        
        Btop = top_layer.B 
        @assert size(Btop) == (K, M)
        @tullio g[k,a] = compute_g(Btop[k,a], ω[k,a], √V[k,a])  avx=false
        @tullio Γ[k,a] := g[k,a] * (ω[k,a] / V[k,a] + g[k,a])

        if !isbottomlayer(layer)
            ## A is used only for continuous actvations
            ## A .= (m.^2 .+ σ)' * Γ .- σ' * g.^2
            # @tullio A[i,a] = m[k,i]^2 * Γ[k,a]
            # @tullio A[i,a] += σ[k,i] * Γ[k,a]
            # @tullio A[i,a] += -σ[k,i] * g[k,a]^2


            @tullio B[i,a] = m[k,i] * g[k,a]
            @tullio B[i,a] += x̂[i,a] * m[k,i]^2 * Γ[k,a]
            # Terms below are 0 for first iteration
            @tullio B[i,a] += -x̂old[i,a] * g[k,a] * gold[k,a] * σ[k,i]
            @tullio B[i,a] += -x̂[i,a] * x̂old[i,a] * σ[k,i] * m[k,i] * gold[k,a] * Γ[k,a]
        end

        if !isfrozen(layer) 
            ## G is used only for continuous weights
            ## G = Γ * (x̂.^2 .+ Δ)' .- g.^2 * Δ'
            # @tullio G[k,i] := Γ[k,a] * x̂[i,a]^2
            # @tullio G[k,i] += Γ[k,a] * Δ[i,a]
            # @tullio G[k,i] += -g[k,a]^2 * Δ[i,a]

            @tullio Hin[k,i] := g[k,a] * x̂[i,a]
            @tullio Hin[k,i] += m[k,i] * x̂[i,a]^2 * Γ[k,a]
            # Terms below are 0 for first iteration
            @tullio Hin[k,i] += -mold[k,i] * g[k,a] * gold[k,a] * Δ[i,a]
            @tullio Hin[k,i] += -m[k,i] * mold[k,i] * gold[k,a] * Γ[k,a] * Δ[i,a] * x̂[i,a]
            
            @tullio Hnew[k,i] := Hin[k,i] + Hext[k,i] + rl * H[k,i] 
            
            # H .= ψ[l] .* H .+ (1-ψ[l]) .* Hnew
            H .= Hnew
        end

        mold .= m
        gold .= g
        x̂old .= x̂
        
        mnew = tanh.(H)
        mnew = ψ[l] .* m .+ (1-ψ[l]) .* mnew .* weight_mask
        Δm = mean(abs.(m .- mnew))
        m .= mnew
        σ .= (1 .- m.^2) .* weight_mask
    end
    
    return Δm
end


function reset_downgoing_messages!(lay::TapLayer)
    lay.B .= 0  
    lay.A .= 0
    lay.g .= 0
    lay.gold .= 0
    lay.mold .= lay.m
    if !isbottomlayer(lay)
        lay.x̂ .= 0
        lay.x̂old .= 0
    end
end

function initrand!(layer::TapLayer)
    @extract layer: K N M weight_mask ϵinit
    @extract layer: x̂  Δ m σ 
    @extract layer: B A ω H  V Hext
    @extract layer: mold
    H .= ϵinit .* randn!(similar(m)) + Hext
    m .= tanh.(H) .* weight_mask
    mold .= m
    σ .= (1 .- m.^2) .* weight_mask
end

function fix_input!(layer::TapLayer, x::AbstractMatrix)
    @extract layer: K N M 
    @extract layer: x̂ Δ m  σ 
    @assert size(x) == size(x̂)
    x̂ .= x
    Δ .= 0
end

function getW(layer::TapLayer)
    m = weight_mean(layer)
    return sign.(m) .* layer.weight_mask
end

function weight_mean(layer::TapLayer)
    # return tanh.(layer.H)
    return layer.m
end

function weight_var(layer::TapLayer)
    m = weight_mean(layer)
    return 1 .- m.^2
end

function forward(layer::TapLayer, x)
    @extract layer: N K
    @assert size(x, 1) == N
    W = getW(layer)
    return sign.(W*x .+ 1f-10)
end

function fixW!(layer::TapLayer, w=1.)
    @extract layer: K N M m σ  weight_mask
    m .= w .* weight_mask
    σ .= 0
end