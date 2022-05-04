using DelimitedFiles, Statistics
using PyPlot
using PyCall
using Printf
using JLD2

"""
# paste the following lines in the REPL and the include("plots_weighthist.jl")
using Pkg
Pkg.activate("../")
#Pkg.activate("./")
Pkg.instantiate()
using DeepMP
"""

rd(x, n) = round(x, sigdigits=n)

plot_bp, plot_sgd = true, true

dataset = :mnist
batchsize = 128
Nin = dataset ≠ :cifar10 ? 784 : 3072
K = [Nin, 101, 101, 1]
L = length(K)-1
density = 1.0
lays = [:bpi]

is = [1]

lrsgd = 0.001
seedgd = 2
classes = []
Ksgd = [101, 101]
P = dataset ≠ :cifar10 ? 6e4 : 5e4
algo = "adam"

fig = plt.figure(constrained_layout=true, figsize=(6.4,4.8))
gs = fig.add_gridspec(1, 1)
ax1 = fig.add_subplot(py"$(gs)[:]")

colors = ["tab:blue", "tab:orange", "tab:green"]
colorsbp = ["tab:cyan", "tab:red", "tab:olive"]

if lays[1] == :bpi
    ρ = [0.9999, 0.9999, 0.9] # Bayesian
    #ρ = [1.0001, 1.0001, 0.9] # point-wise
    binw = true
else
    ρ = [1.0, 1.0, 0.9]
    binw = false
end

if plot_sgd
    file = "../scripts/results_ECE/res_confset$(dataset)_classes$(classes)_binw$(binw)_hidden$(Ksgd)_biasfalse_freezetopfalse"
    (P > 0 && (P≠6e4) && P≠5e4) && (file *= "_P$(Int(P))")
    algo ≠ "adam" && (file *= "_$(algo)")
    file *= "_lr$(lrsgd)_bs$(batchsize)"
    seedgd ≠ 2 && (file *= "_seed$(seedgd)")
    file *= ".jld2"
    @show file    
    if isfile(file)
        weights_sgd = load(file, "weights")
        for i in is
            ax1.hist(vec(weights_sgd[i]), bins=100, density=true, alpha=0.3, 
            color=colors[i], histtype="bar", label="BinaryNet")
            #            color=colors[i], histtype="bar", label="Adam layer $i")
        end
    else
        println("* NOT FOUND: $file")
    end
end


r = [0.0, 0.0, 0.0]
ψ = [0.8, 0.8, 0.8]
ϵinit = 1.0
maxiters = 1
seed = 2


if plot_bp
    resfile = "../scripts/results_ECE/confres_dataset$(dataset)_"
    resfile *= "Ks$(K)_bs$(batchsize)_layers$(lays[1])_rho$(ρ)_r$(r)_damp$(ψ)"
    resfile *= "_density$(density)"
    resfile *= "_M$(Int(P))_ϵinit$(ϵinit)_maxiters$(maxiters)"
    seed ≠ -1 && (resfile *= "_seed$(seed)")
    resfile *= ".jld2"
    if isfile(resfile)
        graph_bp = load(resfile, "graph")
        #weights_bp = DeepMP.getW(graph_bp)
        for i in is
            #magnetizations = graph_bp.layers[i+1].m
            magnetizations = graph_bp
            #magnetizations = DeepMP.getW(graph_bp.layers[i+1])
            ax1.hist(vec(magnetizations), bins=100, density=true, alpha=0.3, 
                    color=colorsbp[i], histtype="bar", label="BP")
                    #color=colorsbp[i], histtype="bar", label="BP layer $i")
        end
    else
        println("* NOT FOUND: $resfile")
    end
end

ax1.set_xlabel("w", fontsize=18)
ax1.set_ylabel("P(w)", fontsize=18)
ax1.legend(loc="best", frameon=false, fontsize=14)

title = binw == true ? "Binary Weights" : "Continuous Weights"
#ax1.set_title(title, fontsize = 16)

fig.savefig("figures/weight_histogram.png")
fig.savefig("figures/weight_histogram.pdf")
plt.close()