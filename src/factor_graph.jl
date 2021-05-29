mutable struct FactorGraph
    K::Vector{Int} # dimension of hidden layers
    M::Int          # number of training examples
    L::Int          # Number of hidden layers. L=length(layers)-2
    layers::Vector{AbstractLayer}  # First and Last Layers are input and output layers
                                   # Weight with layers are those in 2:L+1
    density # weight density (fraction of non-zeros)
    device

    function FactorGraph(x::AbstractMatrix, y::AbstractVector,
                K::Vector{Int}, ϵinit::F,
                layertype::Vector{Symbol};
                β=Inf,
                density=1., verbose=1,
                device=cpu)
        N, M = size(x)
        @assert length(y) == M

        L = length(K)-1
        density = process_density(density, L)
        numW = length(K)==2 ? K[1]*K[2]*density[1]  :
                        sum(l->density[l] * K[l]*K[l+1], 1:length(K)-2)
        numW = round(Int, numW)
        @assert K[1]==N
        verbose > 0 && println("# N=$N M=$M α=$(M/numW) device=$device")

        x, y = x |> device, y |> device

        layers = Vector{AbstractLayer}()
        push!(layers, InputLayer(x))
        # verbose > 0 &&  println("Created InputLayer")

        for l=1:L
            if  layertype[l] == :mf
                push!(layers, MeanFieldLayer(K[l+1], K[l], M, ϵinit, density=density[l]))
                verbose > 0 && println("Created MeanFieldLayer\t $(K[l])")
            elseif  layertype[l] == :tap
                push!(layers, TapLayer(K[l+1], K[l], M, ϵinit, density=density[l]))
                verbose > 0 && println("Created TapLayer\t $(K[l])")
            elseif  layertype[l] == :tapex
                push!(layers, TapExactLayer(K[l+1], K[l], M))
                verbose > 0 && println("Created TapExactLayer\t $(K[l])")
            elseif  layertype[l] == :bp
                push!(layers, BPLayer(K[l+1], K[l], M, ϵinit, density=density[l]))
                verbose > 0 && println("Created BPLayer\t $(K[l])")
            elseif  layertype[l] == :bpacc
                push!(layers, BPAccurateLayer(K[l+1], K[l], M, density=density[l]))
                verbose > 0 && println("Created BPAccurateLayer\t $(K[l])")
            elseif  layertype[l] == :bpex
                push!(layers, BPExactLayer(K[l+1], K[l], M, density=density[l]))
                verbose > 0 && println("Created BPExactLayer\t $(K[l])")
            elseif  layertype[l] == :bpi
                push!(layers, BPILayer(K[l+1], K[l], M, ϵinit, 
                        density=density[l], type=layertype[l]))
                verbose > 0 && println("Created BPILayer\t $(K[l])")
            elseif  layertype[l] == :bpreal
                @assert l == 1
                push!(layers, BPRealLayer(K[l+1], K[l], M))
                verbose > 0 && println("Created BPRealLayer\t $(K[l])")
            else
                error("Wrong Layer Symbol")
            end
        end

        push!(layers, OutputLayer(y, β=β))
        # verbose > 0 && println("Created OutputLayer")

        layers = device.(layers)
        for l=1:L+1
            chain!(layers[l], layers[l+1])
        end

        new(K, M, L, layers, density, device)
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

function has_same_size(g::FactorGraph, W::Vector{<:AbstractMatrix})
    length(g.K) == length(W) + 1 || return false
    for i in 1:length(g.K)-1
       g.K[i] == size(W[i], 2) || return false
    end
    g.K[end] == size(W[end], 1) || return false
    return true
end

function set_weight_mask!(g::FactorGraph, W::Vector{<:AbstractMatrix})
    @assert length(W) == g.L
    for l=1:g.L
        mask = W[l] .!= 0
        set_weight_mask!(g.layers[l+1], g.device(mask))
    end
end

function set_weight_mask!(g::FactorGraph, g2::FactorGraph)
    @assert g2.L == g.L
    for l=2:g.L+1
        set_weight_mask!(g.layers[l], g2.layers[l].weight_mask)
    end
end

function set_external_fields!(layer::AbstractLayer, h0; ρ=1., rbatch=0)
    if hasproperty(layer, :allhext)
        for k = 1:layer.K
            layer.allhext[k] .= ρ .* h0[k] .+ rbatch .* layer.allhext[k]
        end
    else
        layer.Hext .= ρ .* h0 .+ rbatch .* layer.Hext
    end
end

function set_external_fields!(g::FactorGraph, h0; ρ=1.0, rbatch=0)
    @assert length(h0) == g.L
    for l = 2:g.L+1
        set_external_fields!(g.layers[l], h0[l-1]; ρ, rbatch)
    end
end

# set eternal field ffrom posterior
function set_Hext_from_H!(g::FactorGraph, ρ, rbatch)
    for l = 2:g.L+1
        set_Hext_from_H!(g.layers[l], ρ, rbatch)
    end
end

function set_Hext_from_H!(lay::AbstractLayer, ρ, rbatch)
    if hasproperty(lay, :allh) # TODO deprecate
        @assert hasproperty(lay, :allhext)
        for k in 1:lay.K
            lay.allhext[k] .= ρ .* lay.allh[k] .+ rbatch .* lay.allhext[k]
        end
    else
        lay.Hext .= ρ .* lay.H .+ rbatch .* lay.Hext
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

# function copy_allh!(hext, lay::AbstractLayer; ρ=1.0)
#     if hasproperty(lay, :allh)
#         for k in 1:lay.K
#             @assert all(isfinite, lay.allh[k])
#             hext[k] .= ρ .* lay.allh[k]
#         end
#     else
#         hext .= ρ .* lay.H
#     end
# end

# function copy_allh!(hext, g::FactorGraph; ρ=1.0)
#     for l = 2:g.L+1
#         copy_allh!(hext[l-1], g.layers[l]; ρ)
#     end
# end

# function get_allh(layer::AbstractLayer)
#     if hasproperty(layer, :allh)
#         return layer.allh
#     else
#         return layer.H
#     end
# end

# function get_allh(g::FactorGraph)
#     [get_allh(layer) for layer in g.layers[2:g.L+1]]
# end

function initrand!(g::FactorGraph)
    @extract g: M layers K
    for lay in layers[2:end-1]
        initrand!(lay)
    end
    fixY!(g.layers[2], g.layers[1].x) # fix input to first layer
end

function set_input_output!(g, x, y)
    set_output!(g.layers[end], y)
    g.layers[1].x = x
    fixY!(g.layers[2], g.layers[1].x) # fix input to first layer
    
    # Set to 0 the messages going down
    for lay in g.layers[2:end-1]
        lay.B .= 0  
        if hasproperty(lay, :Bcav)
            lay.Bcav .= 0
        end
        if hasproperty(lay, :mcav)
            lay.mcav .= lay.m
        end
        if hasproperty(lay, :g)
            lay.g .= 0
        end
    end
end

function freezetop!(g::FactorGraph, w)
    if g.L != 1
        fixW!(g.layers[end-1], w)
        freeze!(g.layers[end-1])
    end
end

function update!(g::FactorGraph, reinfpar)
    Δ = 0.

    for l = 2:g.L+1
        δ = update!(g.layers[l], reinfpar; mode=:forw)
        Δ = max(δ, Δ)
    end

    for l = (g.L+1):-1:2
        δ = update!(g.layers[l], reinfpar; mode=:back)
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
    x = g.layers[1].x
    y = g.layers[end].y
    ŷ = forward(g, x) |> vec
    return sum(ŷ .!= y)
end

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
