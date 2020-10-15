
mutable struct FactorGraph
    K::Vector{Int} # dimension of hidden layers
    M::Int
    L::Int          # Number of hidden layers. L=length(layers)-2
    ξ::Matrix{Float64}
    σ::Vector{Int}
    layers::Vector{AbstractLayer}  # First and Last Layers are input and output layers
                                   # Weight with layers are those in 2:L+1
    dropout::Dropout
    density # weight density (ONLY FOR bp family as of yet)

    function FactorGraph(ξ::Matrix{Float64}, σ::Vector{Int}
                , K::Vector{Int}, layertype::Vector{Symbol}; β=Inf, βms = 1.,rms =1., ndrops=0,
                density=1., verbose=1)
        N, M = size(ξ)
        @assert length(σ) == M

        L = length(K)-1
        density = process_density(density, L)
        numW = length(K)==2 ? K[1]*K[2]*density[1]  :
            sum(l->density[l] * K[l]*K[l+1], 1:length(K)-2)
        numW = round(Int, numW)
        @assert K[1]==N
        verbose > 0 && println("# N=$N M=$M α=$(M/numW)")

        layers = Vector{AbstractLayer}()
        push!(layers, InputLayer(ξ))
        verbose > 0 &&  println("Created InputLayer")


        for l=1:L
            if  layertype[l] == :tap
                push!(layers, TapLayer(K[l+1], K[l], M, density=density[l]))
                verbose > 0 && println("Created TapLayer\t $(K[l])")
            elseif  layertype[l] == :tapex
                push!(layers, TapExactLayer(K[l+1], K[l], M))
                verbose > 0 && println("Created TapExactLayer\t $(K[l])")
            elseif  layertype[l] == :bp
                push!(layers, BPLayer(K[l+1], K[l], M, density=density[l]))
                verbose > 0 && println("Created BPLayer\t $(K[l])")
            elseif  layertype[l] == :bpacc
                #push!(layers, BPLayer(K[l+1], K[l], M))
                push!(layers, BPAccurateLayer(K[l+1], K[l], M, density=density[l]))
                verbose > 0 && println("Created BPAccurateLayer\t $(K[l])")
            elseif  layertype[l] == :bpex
                push!(layers, BPExactLayer(K[l+1], K[l], M, density=density[l]))
                verbose > 0 && println("Created BPExactLayer\t $(K[l])")
            elseif  layertype[l] == :bpi
                push!(layers, BPILayer(K[l+1], K[l], M, density=density[l]))
                verbose > 0 && println("Created BPILayer\t $(K[l])")
            elseif  layertype[l] == :ms
                push!(layers, MaxSumLayer(K[l+1], K[l], M, βms=βms, rms=rms))
                verbose > 0 && println("Created MaxSumLayer\t $(K[l])")
            elseif  layertype[l] == :parity
                @assert l == L
                push!(layers, ParityLayer(K[l+1], K[l], M))
                verbose > 0 && println("Created ParityLayer\t $(K[l])")
            elseif  layertype[l] == :bpreal
                @assert l == 1
                push!(layers, BPRealLayer(K[l+1], K[l], M))
                verbose > 0 && println("Created BPRealLayer\t $(K[l])")
            else
                error("Wrong Layer Symbol")
            end
        end

        push!(layers, OutputLayer(σ,β=β))
        verbose > 0 && println("Created OutputLayer")

        for l=1:L+1
            chain!(layers[l], layers[l+1])
        end

        dropout = Dropout()
        add_rand_drops!(dropout, 3, K[2], M, ndrops)
        new(K, M, L, ξ, σ, layers, dropout)
    end
end

# Turn density into a vector (a value for each layer)
function process_density(density, L)
    if isa(density, Number)
        density = fill(density, L)
    end
    @assert length(density) == L
    if density[L] < 1.0
        density[L] = 1.0
        # @warn "Setting density[$L] = 1.0"
    end
    return density
end

function set_weight_mask!(g::FactorGraph, W)
    @assert length(W) == g.L
    for l=1:g.L
        K = length(W[l])
        mask = [map(x-> x==0 ? 0 : 1, W[l][k]) for k=1:K]
        set_weight_mask!(g.layers[l+1], mask)
    end
end

function set_weight_mask!(g::FactorGraph, g2::FactorGraph)
    @assert g2.L == g.L
    for l=2:g.L
        set_weight_mask!(g.layers[l], g2.layers[l].weight_mask)
    end
end

function set_external_fields!(g::FactorGraph, h0; ρ=1.0)
    # @extract g K
    # for l = 1:length(K)-1
    #     @assert length(h0[l]) == K[l+1]
    #     for k = 1:K[l+1]
    #         g.layers[l+1].allhext[k] .= ρ .* h0[l][k] .* g.layers[l+1].weight_mask[k]
    #     end
    # end
    for l = 2:g.L+1
        # @assert length(h0[l]) == K[l+1]
        for k = 1:g.layers[l].K
            g.layers[l].allhext[k] .= ρ .* h0[l-1][k] .* g.layers[l].weight_mask[k]
        end
    end
end

function copy_allh(g::FactorGraph, hext; ρ=1.0)
    for l = 2:g.L+1
        for k in 1:g.layers[l].K
            @assert all(isfinite, g.layers[l].allh[k])
            hext[l-1][k] .= ρ .* g.layers[l].allh[k] .* g.layers[l].weight_mask[k]
        end
    end
end

function init_hext(K::Vector{Int}; ϵ=0.0)
    hext = [[ϵ .* rand(K[l]) for k = 1:K[l+1]] for l = 1:length(K)-1]
    return hext
end

function initrand!(g::FactorGraph)
    @extract g M layers K ξ
    for lay in layers[2:end-1]
        initrand!(lay)
    end
end

function fixtopbottom!(g::FactorGraph)
    @extract g M layers K ξ
    if g.L != 1
        # fixW!(g.layers[end-1], 1.)
    end

    fixY!(g.layers[2], ξ)
end

function update!(g::FactorGraph, reinfpar)
    Δ = 0. # Updating layer $(lay.l)")
    # for l=2:g.L+1
    # for l in shuffle([2:g.L+1]...)
    for l = (g.L+1):-1:2
        dropout!(g, l+1)
        δ = update!(g.layers[l], reinfpar)
        Δ = max(δ, Δ)
    end
    return Δ
end

function forward(g::FactorGraph, ξ::Vector)
    @extract g: L layers
    σks = deepcopy(ξ)
   for l=2:L+1
        σks = forward(layers[l], σks)
    end
    return σks
end

function energy(g::FactorGraph)
    @extract g: M ξ
    E = 0
    for a=1:M
        σks = forward(g, ξ[:,a])
        E += energy(g.layers[end], σks, a)
    end

    E
end

mags(g::FactorGraph) = [(lay.allm)::VecVec for lay in g.layers[2:end-1]]

getW(g::FactorGraph) = [getW(lay) for lay in g.layers[2:end-1]]

function dropout!(g::FactorGraph, level::Int)
    @extract g: dropout layers
    !haskey(dropout.drops, level) && return
    pd = layers[level].allpd
    for (k, μ) in dropout.drops[level]
        pd[k][μ] = 0.5
    end
end

function plot_info(g::FactorGraph, info=1; verbose=0, teacher=nothing)
    #W = getW(g)
    K = g.K
    L = length(K)-1
    N = K[1]
    #N = length(W[1][1])
    layers = g.layers[2:end-1]
    @assert length(layers) == L
    width = info
    info > 0 && clf()
    for l=1:L

        q0 = Float64[]
        qWαβ = Float64[]
        R = Float64[]
        for k=1:K[l+1]
            if hasproperty(layers[l], :weight_mask)
                Nk = sum(layers[l].weight_mask[k])
            else
                Nk = K[l]
            end
            push!(q0, dot(layers[l].allm[k], layers[l].allm[k]) / Nk)

            if teacher !== nothing
                @assert length(teacher) == L
                push!(R, dot(layers[l].allm[k], teacher[l][k]) / Nk)
            end
            for p=k+1:K[l+1]
                if hasproperty(layers[l], :weight_mask)
                    Np = sum(layers[l].weight_mask[p])
                else
                    Np = K[l]
                end
                # push!(q, dot(W[l][k],W[l][p])/K[l])
                # push!(qWαβ, dot(layers[l].allm[k], layers[l].allm[p]) / sqrt(q0[k]*q0[p])/K[l])
                push!(qWαβ, dot(layers[l].allm[k], layers[l].allm[p])
                    / sqrt(Nk*Np))
            end
        end

        verbose > 0 && printvec(q0, "layer $l q0=")
        verbose > 0 && printvec(qWαβ, "layer $l qWαβ=")
        verbose > 0 && printvec(R, "layer $l R=")

        info == 0 && continue

        subplot(L,width,width*(L-l)+1)
        title("W Overlaps Layer $l")
        xlim(-1.01,1.01)
        #plt[:hist](q)
        plt.hist(qWαβ)
        info == 1 && continue

        subplot(L,width,width*(L-l)+2)
        title("Mags Layer $l")
        xlim(-1.01,1.01)
        #plt[:hist](vcat(m[l]...))
        plt.hist(vcat(layers[l].allm...))
        info == 2 && continue

        subplot(L,width,width*(L-l)+3)
        title("Fact Satisfaction Layer $l")
        xlim(-1.01,1.01)
        for k=1:K[l+1]
            pu = layers[l].allpu[k]
            pd = layers[l].top_allpd[k]
            #sat = (2pu-1) .* (2pd-1)
            sat = @. (2pu-1) * (2pd-1)
            #plt[:hist](sat)
            #@show size(sat)
            #plt.hist(sat)
        end
        info == 3 && continue

        subplot(L,width,width*(L-l)+4)
        title("Mag UP From Layer $l")
        xlim(-1.01,1.01)
        for k=1:K[l+1]
            pu = layers[l].allpu[k]
            #plt[:hist](2pu-1)
            plt.hist(2 .* pu .- 1)
        end
        info == 4 && continue


        subplot(L,width,width*(L-l)+5)
        title("Mag DOWN To Layer $l")
        xlim(-1.01,1.01)
        for k=1:K[l+1]
            pd = layers[l].top_allpd[k]
            #plt.hist(2 .* pd .- 1)
        end
        info == 5 && continue

        tight_layout()

    end
end
