mutable struct FactorGraph
    K::Vector{Int} # dimension of hidden layers
    M::Int          # number of training examples
    L::Int          # Number of hidden layers. L=length(layers)-2
    x::Matrix{Float64}
    y::Vector{Int}
    layers::Vector{AbstractLayer}  # First and Last Layers are input and output layers
                                   # Weight with layers are those in 2:L+1
    density # weight density (ONLY FOR bp family as of yet)

    function FactorGraph(x::Matrix{Float64}, y::Vector{Int},
                K::Vector{Int},
                layertype::Vector{Symbol};
                β=Inf, βms = 1.,
                density=1., verbose=1)
        N, M = size(x)
        @assert length(y) == M

        L = length(K)-1
        density = process_density(density, L)
        numW = length(K)==2 ? K[1]*K[2]*density[1]  :
            sum(l->density[l] * K[l]*K[l+1], 1:length(K)-2)
        numW = round(Int, numW)
        @assert K[1]==N
        verbose > 0 && println("# N=$N M=$M α=$(M/numW)")

        layers = Vector{AbstractLayer}()
        push!(layers, InputLayer(x))
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
            # elseif  layertype[l] == :bp2
            #     push!(layers, BPLayer(K[l+1], K[l], M, density=density[l]))
            #     verbose > 0 && println("Created BPLayer\t $(K[l])")
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
                push!(layers, MaxSumLayer(K[l+1], K[l], M, βms=βms))
                verbose > 0 && println("Created MaxSumLayer\t $(K[l])")
            elseif  layertype[l] == :bpreal
                @assert l == 1
                push!(layers, BPRealLayer(K[l+1], K[l], M))
                verbose > 0 && println("Created BPRealLayer\t $(K[l])")
            else
                error("Wrong Layer Symbol")
            end
        end

        push!(layers, OutputLayer(y, β=β))
        verbose > 0 && println("Created OutputLayer")

        for l=1:L+1
            chain!(layers[l], layers[l+1])
        end

        new(K, M, L, x, y, layers)
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

function set_weight_mask!(g::FactorGraph, W::VecVecVec)
    @assert length(W) == g.L
    for l=1:g.L
        K = length(W[l])
        N = length(W[l][1])
        w = W[l]
        mask = [w[i][j]==0 ? 0 : 1 for i=1:K,j=1:N]
        set_weight_mask!(g.layers[l+1], mask)
    end
end

function set_weight_mask!(g::FactorGraph, g2::FactorGraph)
    @assert g2.L == g.L
    for l=2:g.L+1
        set_weight_mask!(g.layers[l], g2.layers[l].weight_mask)
    end
end

function set_external_fields!(layer::AbstractLayer, h0; ρ=1.)
    if hasproperty(layer, :allhext)
        for k = 1:layer.K
            layer.allhext[k] .= ρ .* h0[k]
        end
    else
        layer.Hext .= ρ .* h0
    end
end

function set_external_fields!(g::FactorGraph, h0; ρ=1.0)
    @assert length(h0) == g.L
    for l = 2:g.L+1
        set_external_fields!(g.layers[l], h0[l-1]; ρ)
    end
end

function copy_mags!(lay1::AbstractLayer, lay2::AbstractLayer)
    if hasproperty(lay1, :allm)
        for k in 1:lay1.K
            lay1.allm[k] .= lay2.allm[k]
        end
    else
        lay1.m .= lay2.m
    end
end

function copy_mags!(g1::FactorGraph, g2::FactorGraph)
    @assert g1.L == g2.L
    for l = 2:g1.L+1
        copy_mags!(g1.layers[l], g2.layers[l])
    end
end

function copy_allh!(hext, lay::AbstractLayer; ρ=1.0)
    if hasproperty(lay, :allh)
        for k in 1:lay.K
            @assert all(isfinite, lay.allh[k])
            hext[k] .= ρ .* lay.allh[k]
        end
    else
        hext .= ρ .* lay.H
    end
end

function copy_allh!(hext, g::FactorGraph; ρ=1.0)
    for l = 2:g.L+1
        copy_allh!(hext[l-1], g.layers[l]; ρ)
    end
end

function get_allh(layer::AbstractLayer)
    if hasproperty(layer, :allh)
        return layer.allh
    else
        return layer.H
    end
end

function get_allh(g::FactorGraph)
    [get_allh(layer) for layer in g.layers[2:g.L+1]]
end

function init_hext(K::Vector{Int}; ϵ=0.0)
    hext = [[ϵ .* rand(K[l]) for k = 1:K[l+1]] for l = 1:length(K)-1]
    return hext
end

function initrand!(g::FactorGraph)
    @extract g: M layers K x
    for lay in layers[2:end-1]
        initrand!(lay)
    end
    fixY!(g.layers[2], g.x) # fix input to first layer
end

function freezetop!(g::FactorGraph, w)
    if g.L != 1
        fixW!(g.layers[end-1], w)
        freeze!(g.layers[end-1])
    end
end

function update!(g::FactorGraph, reinfpar)
    Δ = 0. # Updating layer $(lay.l)")
    # for l=2:g.L+1
    # for l in shuffle([2:g.L+1]...)
    for l = 2:g.L+1
        δ = update!(g.layers[l], reinfpar)
        Δ = max(δ, Δ)
    end
    for l = (g.L+1):-1:2
        δ = update!(g.layers[l], reinfpar)
        Δ = max(δ, Δ)
    end
    return Δ
end

function forward(g::FactorGraph, x)
    @extract g: L layers
   for l=2:L+1
        x = forward(layers[l], x)
    end
    return x
end

function energy(g::FactorGraph)
    @extract g: x y
    ŷ = forward(g, x) |> vec
    return sum(ŷ .!= y)
end

mags(g::FactorGraph) = [(lay.allm)::VecVec for lay in g.layers[2:end-1]]

getW(g::FactorGraph) = [getW(lay) for lay in g.layers[2:end-1]]

function plot_info(g::FactorGraph, info=1; verbose=0, teacher=nothing)
    K = g.K
    L = length(K)-1
    N = K[1]
    layers = g.layers[2:end-1]
    @assert length(layers) == L
    width = info
    info > 0 && clf()
    for l=1:L
        wt = teacher != nothing ? teacher[l] : nothing
        q0, qWαβ, R = compute_overlaps(layers[l], teacher=wt)


        verbose > 0 && printvec(q0, "layer $l q0=")
        verbose > 0 && printvec(qWαβ, "layer $l qWαβ=")
        verbose > 0 && printvec(R, "layer $l R=")

        info == 0 && continue

        subplot(L, width,width*(L-l)+1)
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
        # TODO (fix this, check if pu and pd are fields and not mags)
        # for k=1:K[l+1]
        #     pu = layers[l].allpu[k]
        #     pd = layers[l].top_allpd[k]
        #     #sat = (2pu-1) .* (2pd-1)
        #     sat = @. (2pu-1) * (2pd-1)
        #     #plt[:hist](sat)
        #     #@show size(sat)
        #     #plt.hist(sat)
        # end
        info == 3 && continue

        subplot(L,width,width*(L-l)+4)
        title("Mag UP From Layer $l")
        xlim(-1.01,1.01)
        # for k=1:K[l+1]
        #     pu = layers[l].allpu[k]
        #     #plt[:hist](2pu-1)
        #     plt.hist(2 .* pu .- 1)
        # end
        info == 4 && continue


        subplot(L,width,width*(L-l)+5)
        title("Mag DOWN To Layer $l")
        xlim(-1.01,1.01)
        # TODO fix
        # for k=1:K[l+1]
        #     pd = layers[l].top_allpd[k]
        #     #plt.hist(2 .* pd .- 1)
        # end
        info == 5 && continue

        tight_layout()

    end
end
