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
            elseif  layertype[l] == :argmax
                push!(layers, ArgmaxLayer(K[l+1], K[l], M, ϵinit, density=density[l]))
                verbose > 0 && println("Created ArgmaxLayer\t $(K[l])")
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
            elseif  layertype[l] == :cbpi
                push!(layers, CBPILayer(K[l+1], K[l], M, ϵinit, 
                        density=density[l], type=layertype[l]))
                verbose > 0 && println("Created CBPILayer\t $(K[l])")
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
        density[L] = 1.0; @warn "Setting density[$L] = 1.0"
    end
    return density
end

# Turn a number into a vector (a value for each layer)
function num_to_vec(v, L)
    if isa(v, Number)
        v = fill(v, L)
    end
    @assert length(v) == L
    return v
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

function write_weight_mask(g::FactorGraph)
    @extract g: density
    for l=2:g.L+1
        file = "results/mask_density$(density)_layer$(l-1).dat"
        writedlm(file, g.layers[l].weight_mask)
        println(file)
    end
end

function set_external_fields!(layer::AbstractLayer, h0; ρ=1., rbatch=0)
    if hasproperty(layer, :allhext)
        # for deprecated tap_exact and bp_exact layers
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

# set eternal field from posterior
function set_Hext_from_H!(g::FactorGraph, ρ, rbatch)
    for l = 2:g.L+1

        # xxx
        # fashion, mnist, cifar10, 101-101
        # first version of argmax
        #ρl = l==2 ? 1.0 :
        #     l==3 ? 1.0 :
        #     l==4 ? 0. : ρ
        # second version of argmax
        #ρl = l==2 ? 0.9999 :
        #     l==3 ? 0.997 :
        #     l==4 ? 0.0 : ρ

        # fashion, mnist, cifar10, 1001-1001
        #ρl = l==2 ? 1.0 :
        #     l==3 ? 1.0 :
        #     l==4 ? 0.0 : ρ

        # batchsize = 1 ; fashion, mnist, cifar10, 101-101
        #ρl = l==2 ? 1.0 :
        #     l==3 ? 1.0 :
        #     l==4 ? 0.999 : ρ

        set_Hext_from_H!(g.layers[l], ρ[l-1], rbatch)
    end
end

f_meta(h; m=0.0) = 1.0 - tanh(m * h)^2 

function set_Hext_from_H!(lay::AbstractLayer, ρ, rbatch)
    lay.Hext .= ρ .* lay.H .+ rbatch .* lay.Hext
    if hasproperty(lay, :Ωext)
        # for continuous weights
        lay.Ωext .= ρ .* lay.Ω .+ rbatch .* lay.Ωext        
    end
end

function initrand!(g::FactorGraph)
    @extract g: M layers K
    for lay in layers[2:end-1]
        initrand!(lay)
    end
    fix_input!(g.layers[2], g.layers[1].x) # fix input to first layer
end

function set_input_output!(g, x, y)
    set_output!(g.layers[end], y)
    g.layers[1].x = x
    fix_input!(g.layers[2], g.layers[1].x) # fix input to first layer
end

function reset_downgoing_messages!(g)
    # Set to 0 the messages going down
    for lay in g.layers[2:end-1]
        reset_downgoing_messages!(lay)
    end
end

function reset_downgoing_messages!(lay::AbstractLayer)
    lay.B .= 0  
    if hasproperty(lay, :Bcav)
        lay.Bcav .= 0
    end
    if hasproperty(lay, :A)
        lay.A .= 0
    end
    if hasproperty(lay, :mcav)
        lay.mcav .= lay.m
    end
    if hasproperty(lay, :g)
        lay.g .= 0
    end
    if hasproperty(lay, :gcav)
        lay.gcav .= 0
    end
    # if hasproperty(lay, :H)
    #     lay.H .= lay.Hext
    # end
end

function freezetop!(g::FactorGraph, w)
    if g.L != 1
        fixW!(g.layers[end-1], w)
        freeze!(g.layers[end-1])
    end
end

function update!(g::FactorGraph, reinfpar)
    Δ = 0.

    # xxx
    # fashion-mnist, mnist, cifar10, 101-101
    #ψ1 = 0.8
    #ψ2 = 0.9
    #ψ3 = 0.9

    for l = 2:g.L+1
        reinfpar.l = l-1
        δ = update!(g.layers[l], reinfpar; mode=:forw)
        Δ = max(δ, Δ)
    end

    for l = (g.L+1):-1:2
        reinfpar.l = l-1
        δ = update!(g.layers[l], reinfpar; mode=:back)
        Δ = max(δ, Δ)
    end

    return Δ
end

"""
Forward pass with pointwise estimator.
"""
function forward(g::FactorGraph, x)
    @extract g: L layers
    for l=2:L+1
        x = forward(layers[l], x)
    end
    return x
end


"""
Forward pass with weight average.
"""
function bayesian_forward(g::FactorGraph, x)
    @extract g: L layers
    x̂ = x 
    Δ = fill!(similar(x), 0)
    for l=2:L+1
        x̂, Δ = bayesian_forward(layers[l], x̂, Δ)
    end
    return x̂, Δ
end

function bayesian_forward(layer::AbstractLayer, x̂, Δ)
    # WARNING Valid only for sign activations
    m = weight_mean(layer) .* layer.weight_mask
    σ = weight_var(layer) .* layer.weight_mask

    @tullio ω[k,a] := m[k,i] * x̂[i,a]
    V = .√(σ * x̂.^2 + m.^2 * Δ + σ * Δ .+ 1f-8)
    @tullio p[k,a] := H(-ω[k,a] / V[k,a]) avx=false
    x̂new = 2p .- 1
    Δnew = 1 .- x̂new.^2 
    return x̂new, Δnew
end


function energy(g::FactorGraph)
    x = g.layers[1].x
    y = g.layers[end].y
    ŷ = forward(g, x) |> vec
    return sum(ŷ .!= y)
end

function bayesian_error(g::FactorGraph, x, y)
    @extract g: K

    ŷ, Δ = bayesian_forward(g, x)

    if K[end] == 1
        ŷ = sign.(ŷ) |> vec
    else
        ŷ = argmax(ŷ, dims=1)
        ŷ = getindex.(ŷ, 1) |> vec    
    end

    #@show size(ŷ) size(y)
    return mean(ŷ .!= y)
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
    Δ0s, q0s, qWαβs = [], [], []
    for l=1:L
        wt = !isnothing(teacher) ? teacher[l] : nothing
        q0, Δ0, qWαβ, R = compute_overlaps(layers[l], teacher=wt)
        
        push!(Δ0s, Δ0)
        push!(q0s, q0)
        push!(qWαβs, qWαβ) 

        verbose > 0 && printvec(q0, "layer $l q0=")
        verbose > 0 && printvec(Δ0, "layer $l Δ0=")
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
    return q0s, qWαβs
end
