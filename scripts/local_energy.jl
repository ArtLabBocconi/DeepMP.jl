using JLD2
using DelimitedFiles
using Random
using MLDatasets
using Statistics
using Knet
using AutoGrad
using Parameters: @with_kw, @unpack
using CUDA
device!(1)

const Flt = Float32

cd("/home/fabrizio/workspace/DeepMP.jl/scripts")

include("/home/fabrizio/workspace/representations/knet/fashion-mnist.jl")
include("/home/fabrizio/workspace/representations/knet/dataset.jl")
include("/home/fabrizio/workspace/representations/knet/utils.jl")

experiment = "bp" # "sgd" or "bp

dataset = "fashion"
batchsize = 128
classes = []
Nin = dataset ≠ "cifar" ? 784 : 3072
K = [Nin, 501, 501, 1]
Ksgd = K[2:end-1]

P = -1
lrsgd = 10.0

if experiment == "sgd"

    seedgd = 2

    file = "../../representations/knet/results/res_conf$(dataset)_classes$(classes)_binwtrue_hidden$(Ksgd)_biasfalse_freezetopfalse"
    (P > 0 && (P≠6e4) && P≠5e4) && (file *= "_P$(Int(P))")
    file *= "_lr$(lrsgd)_bs$(batchsize)"
    seedgd ≠ 2 && (file *= "_seed$(seedgd)")
    file *= ".jld2"

    @show file
    w = load(file)["weights"]
    for l in 1:length(w)
        w[l] .= sign.(w[l])
    end

elseif experiment == "bp"

    P = dataset ≠ :cifar10 ? Int(6e4) : Int(5e4)
    lay = :mf
    ψ = [0.8, 0.8, 0.8]
    if lay ≠ :mf
        ρ = [1.0+1e-4, 1.0+1e-4, 0.9]
    else
        ρ = [1.0+1e-4, 1.0+1e-4, 0.]
    end
    r = 0.0
    density = 1.0
    ϵinit = 1.0
    maxiters = 1
    seed = 2

    multiclass = false
    if !multiclass
        layers = [lay for i in 1:(length(K)-1)]
        classes = []
    else
        K[end] = 10
        layers = [[lay for i in 1:(length(K)-2)]..., :argmax]
    end

    file = "../scripts/results/conf_dataset$(dataset)_"
    file *= "Ks$(K)_bs$(batchsize)_layers$(layers)_rho$(ρ)_r$(r)_damp$(ψ)"
    file *= "_density$(density)"
    file *= "_M$(Int(P))_ϵinit$(ϵinit)_maxiters$(maxiters)"
    seed ≠ -1 && (file *= "_seed$(seed)")
    file *= ".jld2"

    @show file
    w = load(file)["weights"]

end

#w = convert(Array{Float32}, w)

function predict(w, x; l=0)
    x = mat(x)
    for i=1:length(w)
        l == 1 && i > 1 && continue
        ν = √size(w[i], 2)
        x = w[i] * x
        if i < length(w)
            #x = sign.(x./ν)
            x = sign.(x)
        end
    end
    ν = √size(w[end], 2)
    #return x./ν
    return x
end

function binary_accuracy(dtrn, w)
    acc = 0.
    nb = 0
    for (x, y) in dtrn
        scores = convert(Array, sign.(vec(predict(w, x))))
        nb += 1
        acc += mean(scores .== y)
    end
    acc /= nb
    return acc
end

Nin = dataset in ["mnist", "fashion"] ? 28*28 :
dataset == "cifar" ? 32*32*3 : error("Unknown dataset $dataset")
ncolors = dataset in ["cifar"] ? 3 : 1
dtrn, dtrn_acc, dtst_acc = get_data(Nin, ncolors, dataset, P, classes, batchsize, Array{Float32}, false, false, -1)

train_energy, train_energy_std = [1.0 - binary_accuracy(dtrn_acc, w)], [0.0]
println("p: 0.0, local_energy: $(rd(train_energy[1],3)) ± 0.0")

seed = 2
seed > 0 && Random.seed!(seed)

ps = [0.05:0.05:0.35;]
n_stat = 10

for p in ps
    errs = []
    #noise = [randn!(deepcopy(w[1])) for _=1:n_stat]
    for i in 1:n_stat
        wcopy = deepcopy(w)
        for l in 1:length(wcopy)
            noise = rand(size(wcopy[l])[1], size(wcopy[l])[2]) .> p
            #@show noise
            #error()
            noise = 2.0 .* noise .- 1.0
            wcopy[l] .*= noise
        end
        err = 1.0 - binary_accuracy(dtrn_acc, wcopy)
        push!(errs, err)
    end
    mean_err, std_err = mean(errs), std(errs)
    push!(train_energy, mean_err)
    push!(train_energy_std, std_err)
    println("p: $(p), local_energy: $(rd(mean_err,3)) ± $(rd(std_err,3))")
end

writedlm("results/localenergy_dataset$(dataset)_K$(Ksgd)_exp$(experiment).dat", [[0;ps] train_energy train_energy_std])
