
mutable struct BPLayer{A2<:AbstractMatrix,A3<:AbstractArray,M, ACT<:AbstractChannel} <: AbstractLayer
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

    act::ACT
end

@functor BPLayer


function BPLayer(K::Int, N::Int, M::Int, ϵinit; 
                 density=1., isfrozen=false, act=nothing)
    
    x̂ = zeros(F, N, M)
    x̂cav = zeros(F, K, N, M)
    Δ = zeros(F, N, M)
    
    m = zeros(F, K, N)
    mcav = zeros(F, K, N, M)
    σ = zeros(F, K, N)
    
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

    if act === nothing
        act = :sign
    end
    act = channel(act)

    return BPLayer(-1, K, N, M, ϵinit,
            x̂, x̂cav, Δ, 
            m, mcav, σ,
            B, Bcav, A, 
            H, Hext, Hcav,
            ω, ωcav, V,
            DummyLayer(), DummyLayer(),
            weight_mask, isfrozen, 
            act)
end

function Base.show(io::IO, l::BPLayer)
    print(io, "BPLayer($(l.N) => $(l.K), M=$(l.M), act=$(l.act))")
end

function update!(layer::BPLayer, reinfpar; mode=:both)
    @extract layer: K N M weight_mask
    @extract layer: x̂ x̂cav Δ m mcav σ 
    @extract layer: Bup B Bcav A H Hext Hcav ω ωcav V
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
        
        @tullio ω[k,a] = mcav[k,i,a] * x̂cav[k,i,a]
        @tullio ωcav[k,i,a] = ω[k,a] - mcav[k,i,a] * x̂cav[k,i,a]
        V .= σ * x̂.^2 + m.^2 * Δ + σ * Δ .+ 1f-8
    end

    if mode == :back || mode == :both
        ## BACKWARD 
        Btop = top_layer.B 
        Atop = top_layer.A
        @assert size(Btop) == (K, M)
        # @assert all(isfinite, Btop)

        gcav = compute_gcav(act, Btop, Atop, ωcav, V)
        # g = compute_g(act, Btop, Atop, ω, V)
        
        # @tullio Γ[k,a] := compute_Γ(Btop[k,a], ω[k,a], V[k,a])
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
            
            # reinforcement
            @tullio Hnew[k,i] := Hin[k,i] + rl * H[k,i] + Hext[k,i]
        
            @tullio Hcavnew[k,i,a] := Hnew[k,i] - gcav[k,i,a] * x̂cav[k,i,a]
            # H .= ψ[l] .* H .+ (1 - ψ[l]) .* Hnew 
            # Hcav .= ψ[l] .* Hcav .+ (1 - ψ[l]) .* Hcavnew 
            H .= Hnew 
            Hcav .= Hcavnew 
            
            @tullio mcavnew[k,i,a] := tanh(Hcav[k,i,a]) * weight_mask[k,i]
            mcav .= ψ[l] .* mcav .+ (1 - ψ[l]) .* mcavnew
            # mcav .= mcavnew
            
            # @assert all(isfinite, H)
            # @assert all(isfinite, Hcav)


            mnew = ψ[l] .* m .+ (1-ψ[l]) .* tanh.(H) .* weight_mask
            # mnew .= tanh.(H) .* weight_mask
            Δm = mean(abs.(m .- mnew))
            m .= mnew
            σ .= (1 .- m.^2) .* weight_mask    
            @assert all(isfinite, m)
        end
        
    end
    
    return Δm
end

function initrand!(layer::L) where {L <: Union{BPLayer}}
    @extract layer: K N M weight_mask ϵinit
    @extract layer: x̂ x̂cav Δ m mcav σ Hext
    @extract layer: B Bcav A ω H Hcav ωcav V
    # TODO reset all variables
    H .= ϵinit .* randn!(similar(Hext)) + Hext
    m .= tanh.(H) .* weight_mask
    mcav .= m .* weight_mask 
    σ .= (1 .- m.^2) .* weight_mask
end

function fix_input!(layer::L, x::AbstractMatrix) where {L <: Union{BPLayer}}
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
    return layer.act(W*x .+ 1f-10)
end

function fixW!(layer::L, w=1.) where {L <: Union{BPLayer}}
    @extract layer: K N M m σ mcav weight_mask
    m .= w .* weight_mask
    mcav .= m .* weight_mask
    σ .= 0
end
