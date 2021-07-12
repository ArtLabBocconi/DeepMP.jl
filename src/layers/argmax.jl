mutable struct ArgmaxLayer <: AbstractLayer
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
    ω 
    V

    type::Symbol
    top_layer::AbstractLayer
    bottom_layer::AbstractLayer
    weight_mask
    isfrozen::Bool
end

@functor ArgmaxLayer

function ArgmaxLayer(K::Int, N::Int, M::Int, ϵinit; 
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

    return ArgmaxLayer(-1, K, N, M, ϵinit,
            x̂, Δ, m, σ,
            Bup, B, A, 
            H, Hext,
            ω, V,
            type,
            DummyLayer(), DummyLayer(),
            weight_mask, isfrozen)
end

function compute_g_argmax(y, ω, V)
    # transform y (vector of integers) to 2d array of CartesianIndex
    yc = map(t -> CartesianIndex(t[2], t[1]), enumerate(y))
    yc = reshape(yc, 1, :)
    Vtot = .√(V .+ V[yc])
    dω = ω[yc] .- ω
    g = @. -GH(-dω / Vtot) / Vtot 
    g[yc] .= .- sum(g, dims=1) .+ g[yc]
    return g
end

# # version 2 with sampling
# function compute_g_argmax(y, ω, V, nsamples=10)
#     # transform y (vector of integers) to 2d array of CartesianIndex
#     yc = map(t -> CartesianIndex(t[2], t[1]), enumerate(y))
#     yc = reshape(yc, 1, :)
#     V = .√V
#     # @assert size(ω) == (10, 128) 
#     ωc = ω[yc]
#     Vc = V[yc]
#     # @assert size(ωc) == (1, 128)
#     dω = ωc .- ω
#     # @assert size(dω) == (10, 128)
#     g = fill!(similar(ω), 0)
#     # @assert size(g) == (10, 128)
#     for _ in 1:nsamples
#         z = Vc .* randn!(similar(ωc))
#         # @assert size(z) == (1, 128)
#         g .+= @. -GH(-(dω + z) / V) / V 
#     end
#     g[yc] .= .- sum(g, dims=1) .+ g[yc]
#     g ./= nsamples
#     return g
# end

function update!(layer::ArgmaxLayer, reinfpar; mode=:both)
    @extract layer: K N M weight_mask
    @extract layer: x̂ Δ m  σ 
    @extract layer: Bup B A H Hext ω  V
    @extract layer: bottom_layer top_layer
    @extract reinfpar: r y ψ
    Δm = 0.

    # xxx
    #ψargm = 0.9

    if mode == :forw || mode == :both
        if !isbottomlayer(layer)
            bottBup = bottom_layer.Bup
            @tullio x̂[i,a] = tanh(bottBup[i,a] + B[i,a])
            Δ .= 1 .- x̂.^2
        end
        
        @tullio ω[k,a] = m[k,i] * x̂[i,a]
        V .= σ * x̂.^2 + m.^2 * Δ + σ * Δ .+ 1f-8
    end
    if mode == :back || mode == :both
        ytrue = top_layer.y
        @assert size(ytrue) == (M,)
        # @tuillo ωcav[k,i,a] := ω[k,a]- m[k,i] * x̂[i,a]
        g = compute_g_argmax(ytrue, ω, V)
        # @tullio gcav[k,i,a] := g[k,a]
        # @tullio gcav[k,i,a] := compute_g(Btop[k,a], ω[k,a]- m[k,i] * x̂[i,a], V[k,a])  avx=false
        # @tullio Γ[k,a] := compute_Γ(Btop[k,a], ω[k,a], V[k,a])
        
        if !isbottomlayer(layer)
            # @tullio B[i,a] = m[k,i] * gcav[k,i,a]
            @tullio B[i,a] = m[k,i] * g[k,a]
        end

        if !isfrozen(layer)
            # @tullio Hin[k,i] := gcav[k,i,a] * x̂[i,a]
            @tullio Hin[k,i] := g[k,a] * x̂[i,a]
            if y > 0 # focusing
                tγ = tanh(r)
                @tullio mjs[k,i] := tanh(Hin[k,i])
                @tullio mfoc[k,i] := tanh((y-1)*atanh(mjs[k,i]*tγ)) * tγ
                @tullio Hfoc[k,i] := atanh(mfoc[k,i])
                @tullio Hnew[k,i] := Hin[k,i] + Hfoc[k,i] + Hext[k,i] 
            else
                # reinforcement 
                @tullio Hnew[k,i] := Hin[k,i] + r*H[k,i] + Hext[k,i]
            end
            #H .= ψargm .* H .+ (1-ψargm) .* Hnew
            H .= ψ .* H .+ (1-ψ) .* Hnew

            mnew = tanh.(H) .* weight_mask
            Δm = mean(abs.(m .- mnew))
            m .= mnew
            σ .= (1 .- m.^2) .* weight_mask
            # @assert all(isfinite, m)
        end
    end
    
    return Δm
end

function initrand!(layer::L) where {L <: Union{ArgmaxLayer}}
    @extract layer: K N M weight_mask ϵinit
    @extract layer: x̂ Δ m σ 
    @extract layer: B A ω H V Hext
    # TODO reset all variables
    H .= ϵinit .* randn!(similar(m)) + Hext
    m .= tanh.(H) .* weight_mask
    σ .= (1 .- m.^2) .* weight_mask
end

function fix_input!(layer::L, x::AbstractMatrix) where {L <: Union{ArgmaxLayer}}
    @extract layer: K N M 
    @extract layer: x̂ Δ m σ 
    @assert size(x) == size(x̂)
    x̂ .= x
    Δ .= 0
end

function getW(layer::ArgmaxLayer)
    return sign.(layer.m) .* layer.weight_mask
end

function forward(layer::ArgmaxLayer, x)
    @extract layer: N K
    @assert size(x, 1) == N
    W = getW(layer)
    y = argmax(W*x, dims=1)
    return getindex.(y, 1)
end

function fixW!(layer::ArgmaxLayer, w=1.)
    @extract layer: K N M m σ weight_mask
    m .= w .* weight_mask
    σ .= 0
end

