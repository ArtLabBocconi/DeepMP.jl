using DelimitedFiles, Statistics
using PyPlot
using PyCall
using Printf

plt.style.use("default")
plt.style.use("seaborn-whitegrid")
cd("/home/fabrizio/workspace/DeepMP.jl/notebooks")

rd(x, n) = round(x, sigdigits=n)

dataset = :fashion
batchsize = 128
Nin = dataset ≠ :cifar10 ? 784 : 3072
K = [Nin, 101, 101, 1]
L = length(K)-1
plot_sgd = true
lrsgd = 1.0

# for different file names
#lays = [:bp, :bpi, :tap, :mf]
#lays = [:bpi, :tap, :mf]
lays = [:bpi]

final_params = true
multiclass = true

if !multiclass
    if batchsize == 128
        seed_bp = [2]
        seed_sgd = [2]
        ρ1 = [1e-4, 1e-4, 1e-4]  
        P = 6e4         
        maxiters = 1   
        r = 0.        
        #ρs = [ρ1.+1. for _=1:length(lays)]    
        ϵinit = 2.0
        ψ = [0.8, 0.8, 0.8]         
        ρs = [[0.0, 0.0, 0.0].+1.0]    
    end
else
    K[end] = 10
    if batchsize == 128
        seed_bp = [2]
        seed_sgd = [2]
        P = 6e4         
        maxiters = 1   
        r = 0.        
        ϵinit = 2.0
        ψ = [0.8, 0.9, 0.9]         
        ρs = [[0.9999, 0.997, 0.0]]    
    end
end

density = 1.

algo_color = Dict(:sgd=>"black", :bp=>"tab:red", :tap=>"tab:green", :bpi=>"tab:red", :mf=>"tab:orange")
algo_mark = Dict(:sgd=>"o", :bp=>"^", :tap=>"s", :bpi=>"x", :mf=>"D")
errev = 10

# FIGURE 1
fig = plt.figure(constrained_layout=true, figsize=(6.4*1.7,4.8*1.1))
gs = fig.add_gridspec(3, 4)
ax1 = fig.add_subplot(py"$(gs)[:, 0:2]")
ax2 = fig.add_subplot(py"$(gs)[0, 2]")
ax3 = fig.add_subplot(py"$(gs)[0, 3]")
ax4 = fig.add_subplot(py"$(gs)[1, 2]")
ax5 = fig.add_subplot(py"$(gs)[1, 3]")
ax6 = fig.add_subplot(py"$(gs)[2, 2]")
ax7 = fig.add_subplot(py"$(gs)[2, 3]")

for (i,(lay, ρ)) in enumerate(zip(lays, ρs))
        
    if !multiclass
        layers = [lay for i in 1:(length(K)-1)]
    else
        layers = [[lay for i in 1:(length(K)-2)]..., :argmax]
    end
    
    epoche_bp, train_bp, test_bp = [],[], []
    q0lay1, qablay1 = [], []
    q0lay2, qablay2 = [], []
    q0lay3, qablay3 = [], []

    for seed in seed_bp
        resfile = "../scripts/results/res_dataset$(dataset)_"
        resfile *= "Ks$(K)_bs$(batchsize)_layers$(layers)_rho$(ρ)_r$(r)_damp$(ψ)"
        resfile *= "_density$(density)"
        resfile *= "_M$(Int(P))_ϵinit$(ϵinit)_maxiters$(maxiters)"
        seed ≠ -1 && (resfile *= "_seed$(seed)")
        resfile *= ".dat"
        @show resfile

        dati = readdlm(resfile)

        push!(epoche_bp, dati[:, 1])
        push!(train_bp, dati[:, 2])
        push!(test_bp, dati[:, 3])
        push!(q0lay1, dati[:, 4])
        push!(qablay1, dati[:, 5])
        push!(q0lay2, dati[:, 6])
        push!(qablay2, dati[:, 7])
        push!(q0lay3, dati[:, 8])
        push!(qablay3, dati[:, 9])

    end

    μ_train_bp, σ_train_bp = mean(train_bp), std(train_bp)
    μ_test_bp, σ_test_bp = mean(test_bp), std(test_bp)

    μ_q0lay1, σ_q0lay1 = mean(q0lay1), std(q0lay1)
    μ_qablay1, σ_qablay1 = mean(qablay1), std(qablay1)
    μ_q0lay2, σ_q0lay2 = mean(q0lay2), std(q0lay2)
    μ_qablay2, σ_qablay2 = mean(qablay2), std(qablay2)
    μ_q0lay3, σ_q0lay3 = mean(q0lay3), std(q0lay3)
    μ_qablay3, σ_qablay3 = mean(qablay3), std(qablay3)

    pars = "ρ=$([rd(ρ[l]-1,1) for l=1:L])"

    ax1.errorbar(epoche_bp[1], μ_train_bp, σ_train_bp, ls="-", errorevery=errev,
                 label="train $lay $pars", c=algo_color[lay])
    ax1.errorbar(epoche_bp[1], μ_test_bp, σ_test_bp, ls="--", errorevery=errev,
                 label="test $lay $pars", c=algo_color[lay])

    #ax1.set_xlabel("epochs", fontsize=12)
    ax1.set_ylabel("error (%)", fontsize=12)
    if dataset ≠ :cifar10
        ax1.set_ylim(0,30)
    else
        ax1.set_ylim(20,90)
    end

    ax2.errorbar(epoche_bp[1], μ_q0lay1, σ_q0lay1, ls="-", errorevery=errev,
                 label="$lay lay1", c=algo_color[lay])
    ax3.errorbar(epoche_bp[1], μ_qablay1, σ_qablay1, ls="-", errorevery=errev,
                 label="$lay lay1", c=algo_color[lay])
    ax4.errorbar(epoche_bp[1], μ_q0lay2, σ_q0lay2, ls="-", errorevery=errev,
                 label="$lay lay2", c=algo_color[lay])
    ax5.errorbar(epoche_bp[1], μ_qablay2, σ_qablay2, ls="-", errorevery=errev,
                 label="$lay lay2", c=algo_color[lay])
    ax6.errorbar(epoche_bp[1], μ_q0lay3, σ_q0lay3, ls="-", errorevery=errev,
                 label="$lay lay3", c=algo_color[lay])
    ax7.errorbar(epoche_bp[1], μ_qablay3, σ_qablay3, ls="-", errorevery=errev,
                 label="$lay lay3", c=algo_color[lay])

end

Ksgd = K[2:end-1]
classes = multiclass ? nothing : []
dset_sgd = dataset==:cifar10 ? :cifar : dataset

if plot_sgd
    epoche, train_sgd, test_sgd = [],[], []
    for seedgd in seed_sgd
        file = "../../representations/knet/results/res_dataset$(dset_sgd)_classes$(classes)_binwtrue_hidden$(Ksgd)_biasfalse_freezetopfalse"
        (P > 0 && (P≠6e4 && bs≠600) && P≠5e4 ) && (file *= "_P$(Int(P))")
        file *= "_lr$(lrsgd)_bs$(batchsize)"
        seedgd ≠ 2 && (file *= "_seed$(seedgd)")
        file *= ".dat"
        @show file

        dati_sgd = readdlm(file)

        push!(epoche, dati_sgd[:, 1])
        push!(train_sgd, dati_sgd[:, 2])
        push!(test_sgd, dati_sgd[:, 3])

    end

    μ_train, σ_train = mean(train_sgd) .* 100., std(train_sgd) .* 100.
    μ_test, σ_test = mean(test_sgd) .* 100., std(test_sgd) .* 100.

    ax1.errorbar(epoche[1], μ_train, σ_train, ls="-", c=algo_color[:sgd], errorevery=errev,
                 capsize=0, label="train bin-sgd bs=$batchsize, lr=$lrsgd")
    ax1.errorbar(epoche[1], μ_test, σ_test, ls="--", ms=1, c=algo_color[:sgd], errorevery=errev,
                 capsize=0, label="test bin-sgd bs=$batchsize, lr=$lrsgd")
end

ax1.set_xlabel("epochs", fontsize=12)

ax2.set_ylabel("q0", fontsize=10)
ax2.set_xlabel("epochs", fontsize=10)
ax3.set_xlabel("epochs", fontsize=10)
ax3.set_ylabel("qab", fontsize=10)
ax4.set_ylabel("q0", fontsize=10)
ax4.set_xlabel("epochs", fontsize=10)
ax5.set_xlabel("epochs", fontsize=10)
ax5.set_ylabel("qab", fontsize=10)
ax6.set_ylabel("q0", fontsize=10)
ax6.set_xlabel("epochs", fontsize=10)
ax7.set_xlabel("epochs", fontsize=10)
ax7.set_ylabel("qab", fontsize=10)

#ax3.tick_params(labelsize=7)
#ax2.set_ylim(0,1)

ax1.legend(loc="upper right", frameon=false, fontsize=12)
ax2.legend(loc="best", frameon=false, fontsize=10)
ax3.legend(loc="best", frameon=false, fontsize=10)
ax4.legend(loc="best", frameon=false, fontsize=10)
ax5.legend(loc="best", frameon=false, fontsize=10)
ax6.legend(loc="best", frameon=false, fontsize=10)
ax7.legend(loc="best", frameon=false, fontsize=10)

#plt.grid(false)

#plt.subplots_adjust(hspace=0.2, wspace=0.4)

classt = multiclass ? "10class" : "2class"
Pstring = "$P"[1] * "e$(length("$(Int(P))")-1)"
dset_tit = dataset == :mnist ? "MNIST" :
           dataset == :fashion ? "FashionMNIST" :
           dataset == :cifar ? "CIFAR10" : "?"
fig.suptitle("$dset_tit $classt P=$(Pstring), bs=$batchsize, K=$(K[2:end-1]), ψ=$ψ, init=$ϵinit, iters=$maxiters, r=$r")
#fig.tight_layout()

#fig.savefig("figures/deepMP_bs$(batchsize)_K$(K)_rho$(ρ1)_ψ_$(ψ)_P$(P)_maxiters_$(maxiters)_r$(r)_ϵinit_$(ϵinit)_.png")
fig.savefig("figures/figure_deepMP.png")

plt.close()

## FIGURE 2
#nlays = length(K)-1
#fig, ax = plt.subplots(nlays,2)
#
#for (i,(lay, ρ)) in enumerate(zip(lays, ρs))
#    
#    if !multiclass
#        layers = [lay for i in 1:(length(K)-1)]
#    else
#        layers = [[lay for i in 1:(length(K)-2)]..., :argmax]
#    end
#    
#    resfile = "../scripts/results/res_dataset$(dataset)_"
#    resfile *= "Ks$(K)_bs$(batchsize)_layers$(layers)_rho$(ρ)_r$(r)_damp$(ψ)"
#    resfile *= "_density$(density)"
#    resfile *= "_M$(Int(P))_ϵinit$(ϵinit)_maxiters$(maxiters)"
#    seed ≠ -1 && (resfile *= "_seed$(seed)")
#    resfile *= ".dat"
#    
#    @show resfile
#
#    dati = readdlm(resfile)
#
#    ax1.plot(dati[:,1], dati[:,4], ls="-", label="q0 lay1 $lay", c=algo_color[lay])
#    ax[1+nlays].plot(dati[:,1], dati[:,5], ls="-", label="qab lay1 $lay", c=algo_color[lay])
#
#end
#
#ax1.legend(loc="best", frameon=false, fontsize=10)
#ax[1+nlays].legend(loc="best", frameon=false, fontsize=10)
#
#fig.savefig("figures/figure_deepMP2.png")
#
#plt.close()
