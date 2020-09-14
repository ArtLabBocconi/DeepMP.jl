
mutable struct FactorGraph
    K::Vector{Int} # dimension of hidden layers
    M::Int
    L::Int         # number of hidden layers. L=length(layers)-2
    ξ::Matrix{Float64}
    σ::Vector{Int}
    layers::Vector{AbstractLayer}
    dropout::Dropout
    density # weight density (ONLY FOR bp family as of yet)

    function FactorGraph(ξ::Matrix{Float64}, σ::Vector{Int}
                , K::Vector{Int}, layertype::Vector{Symbol}; β=Inf, βms = 1.,rms =1., ndrops=0,
                density=1.)
        N, M = size(ξ)
        @assert length(σ) == M
        println("# N=$N M=$M α=$(M/N)")
        @assert K[1]==N
        L = length(K)-1
        layers = Vector{AbstractLayer}()
        push!(layers, InputLayer(ξ))
        println("Created InputLayer")
        
        if isa(density, Number)
            density = fill(density, L)
            density[L] = 1
        end
        @assert density[L] == 1
        @assert length(density) == L
        
        for l=1:L
            if  layertype[l] == :tap
                push!(layers, TapLayer(K[l+1], K[l], M, density=density[l]))
                println("Created TapLayer\t $(K[l])")
            elseif  layertype[l] == :tapex
                push!(layers, TapExactLayer(K[l+1], K[l], M))
                println("Created TapExactLayer\t $(K[l])")
            elseif  layertype[l] == :bp
                push!(layers, BPLayer(K[l+1], K[l], M, density=density[l]))
                println("Created BPLayer\t $(K[l])")
            elseif  layertype[l] == :bpacc
                #push!(layers, BPLayer(K[l+1], K[l], M))
                push!(layers, BPAccurateLayer(K[l+1], K[l], M, density=density[l]))
                println("Created BPAccurateLayer\t $(K[l])")
            elseif  layertype[l] == :bpex
                push!(layers, BPExactLayer(K[l+1], K[l], M, density=density[l]))
                println("Created BPExactLayer\t $(K[l])")
            elseif  layertype[l] == :bpi
                push!(layers, BPILayer(K[l+1], K[l], M, density=density[l]))
                println("Created BPILayer\t $(K[l])")
            elseif  layertype[l] == :ms
                push!(layers, MaxSumLayer(K[l+1], K[l], M, βms=βms, rms=rms))
                println("Created MaxSumLayer\t $(K[l])")
            elseif  layertype[l] == :parity
                @assert l == L
                push!(layers, ParityLayer(K[l+1], K[l], M))
                println("Created ParityLayer\t $(K[l])")
            elseif  layertype[l] == :bpreal
                @assert l == 1
                push!(layers, BPRealLayer(K[l+1], K[l], M))
                println("Created BPRealLayer\t $(K[l])")
            else
                error("Wrong Layer Symbol")
            end
        end

        push!(layers, OutputLayer(σ,β=β))
        println("Created OutputLayer")

        for l=1:L+1
            chain!(layers[l], layers[l+1])
        end

        dropout = Dropout()
        add_rand_drops!(dropout, 3, K[2], M, ndrops)
        new(K, M, L, ξ, σ, layers, dropout)
    end
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
        fixW!(g.layers[end-1], 1.)
    end

    fixY!(g.layers[2], ξ)
end



function update!(g::FactorGraph, reinfpar)
    Δ = 0. # Updating layer $(lay.l)")
    for l=2:g.L+1
        dropout!(g, l+1)
        δ = update!(g.layers[l], reinfpar)
        Δ = max(δ, Δ)
    end
    return Δ
end


function forward(g::FactorGraph, ξ::Vector)
    @extract g: L layers
    σks = deepcopy(ξ)
    stability = Vec()
    for l=2:L+1
        σks, stability = forward(layers[l], σks)
    end
    return σks, stability
end

function energy(g::FactorGraph)
    @extract g: M ξ
    E = 0
    stability = zeros(M)
    for a=1:M
        σks, stab = forward(g, ξ[:,a])
        stability[a] = sum(stab)
        E += energy(g.layers[end], σks, a)
    end

    E, stability
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

function plot_info(g::FactorGraph, info=1; verbose=0)
    #W = getW(g)
    K = g.K
    L = length(K)-1
    N = K[1]
    #N = length(W[1][1])
    layers = g.layers[2:end-1]
    width = info
    info > 0 && clf()
    for l=1:L
        q0 = Float64[]
        for k=1:K[l+1]
            push!(q0, dot(layers[l].allm[k], layers[l].allm[k])/K[l])
        end
        qWαβ = Float64[]
        for k=1:K[l+1]
            for p=k+1:K[l+1]
                # push!(q, dot(W[l][k],W[l][p])/K[l])
                push!(qWαβ, dot(layers[l].allm[k],layers[l].allm[p]) / sqrt(q0[k]*q0[p])/K[l])
            end
        end
        verbose > 0 && printvec(q0,"layer $l q0=")
        verbose > 0 && printvec(qWαβ,"layer $l qWαβ=")

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