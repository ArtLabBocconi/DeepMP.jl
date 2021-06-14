using DelimitedFiles, Statistics
using PyPlot
using Printf

plt.style.use("default")
plt.style.use("seaborn-whitegrid")
cd("/home/fabrizio/workspace/DeepMP.jl/notebooks")

rd(x, n) = round(x, sigdigits=n)

dataset = :cifar10
batchsize = 128
Nin = dataset ≠ :cifar10 ? 784 : 3072
K = [Nin, 101, 101, 10]
plot_sgd = true
lrsgd = 0.5

# for different file names
#lays = [:bp, :bpi, :tap, :mf]
lays = [:bpi, :tap, :mf]
lays = [:bpi]

final_params = false
multiclass = true
bs = 0

if multiclass
    if batchsize == 128 && K[2] == 1001
        ρ1 = 1e-3
    else
        ρ1 = 0.
    end
    ρs = [ρ1, ρ1, ρ1] .+ 1.
    ψ = 0.9
    maxiters = 1
    r = 0.
    P = dataset ≠ :cifar10 ? 6e4 : 5e4
    if K[2]==101
        seed = 2
    else
        seed = -1
    end
    if batchsize == 128
        ϵinit = 2.
        seed_sgd = [2, 5, 11]
    else
        ϵinit = 0.5
        seed_sgd = [5]
    end
elseif !final_params
    K = [28*28, 101, 101, 1] # [[28*28, 1/5/10-01, (1/5/10-01), (1/5/10-01), 1]]
    ρs = [-1e-1, -1e-5, 0., 1e-6, 1e-5, 1e-4, 1e-3, 1e-2] # saveres=false, ψ=0.5
    ρ1 = 1e-5
    ρs = [ρ1 for _=1:length(lays)] .+ 1.
    ρs = [ρ1, ρ1, ρ1] .+ 1.
    ϵinits = [0., 0.01, 0.1, 0.5, 1., 1.5, 2., 3.]
    ϵinit = 1.
    ψs = [[0:0.2:0.8;]..., 0.9, 0.99, 0.999, 0.9999]
    ψ = 0.999
    maxiters = 1    # 1, 10, 50, 100 # saveres = true, ϵinit = 0 (non va bene sto valore)
    r = 0.          # [0:0.2:1.2;] (for maxiters=10) # saveres = true
    P = 6e4
    batchsizes = [1, 16, 128, 1024] # saveres = false, ψ=0.5
    batchsize = 128
    #P = 1e3; batchsize = Int(P/1e2) # 1e2, 1e3, 1e4, 6e4 (bs = 1e0, 1e1, 1e2, 6e2 respectively) # saveres=true
elseif final_params && bs == 1 # parameters for batchsize=1
    batchsize = 1 
    ρ1 = 1e-6  
    ψ = 0.8    
    P = 6e4       
    maxiters = 1  
    r = 0.        
    ϵinit = 1e0   
    K = [28*28, 501, 501, 501, 1] 
    ρs = [ρ1 for _=1:length(lays)] .+ 1.
elseif final_params && bs == 128 # for beautiful final figure with bs=128
    batchsize = 128
    ρ1 = 1e-5       
    ψ = 0.8         
    P = 6e4         
    maxiters = 1   
    r = 0.        
    ϵinit = 1e0
    K = [28*28, 101, 101, 1]
    ρs = [ρ1 for _=1:length(lays)] .+ 1.
elseif final_params && bs == 0 # for varying architecture, saveres=true
    batchsize = 128
    ρ1 = 1e-4 
    ψ = 0.8         
    P = 6e4         
    maxiters = 1   
    r = 0.        
    ϵinit = 0.5
    K = [28*28, 1001, 1001, 1001, 1]
    ρs = [ρ1 for _=1:length(lays)] .+ 1.
end

density = 1.

algo_color = Dict(:sgd=>"black", :bp=>"tab:red", :tap=>"tab:green", :bpi=>"tab:red", :mf=>"tab:orange")
algo_mark = Dict(:sgd=>"o", :bp=>"^", :tap=>"s", :bpi=>"x", :mf=>"D")

# FIGURE 1
fig, ax1 = plt.subplots(1)
ax2 = ax1.inset_axes([0.18, 0.68, 0.35, 0.3])
if !multiclass || dataset == :mnist
    ax3 = ax1.inset_axes([0.64, 0.33, 0.35, 0.275])
else
    ax3 = ax1.inset_axes([0.1, 0.08, 0.35, 0.275])
end

for (i,(lay, ρ)) in enumerate(zip(lays, ρs))
        
    if !multiclass
        layers = [lay for i in 1:(length(K)-1)]
    else
        layers = [[lay for i in 1:(length(K)-2)]..., :argmax]
    end
    
    resfile = "../scripts/results/res_dataset$(dataset)_"
    resfile *= "Ks$(K)_bs$(batchsize)_layers$(layers)_rho$(ρ)_r$(r)_damp$(ψ)"
    resfile *= "_density$(density)"
    resfile *= "_M$(Int(P))_ϵinit$(ϵinit)_maxiters$(maxiters)"
    seed ≠ -1 && (resfile *= "_seed$(seed)")
    resfile *= ".dat"
    
    @show resfile

    dati = readdlm(resfile)

    pars = "ρ=$(rd(ρ-1,1))"

    ax1.plot(dati[:,1], dati[:,2], ls="-", label="train $lay $pars", c=algo_color[lay])
    ax1.plot(dati[:,1], dati[:,3], ls="--", label="test $lay $pars", c=algo_color[lay])

    #ax1.set_xlabel("epochs", fontsize=12)
    ax1.set_ylabel("error (%)", fontsize=12)
    if dataset ≠ :cifar10
        ax1.set_ylim(0,30)
    else
        ax1.set_ylim(20,90)
    end

    ax2.plot(dati[:,1], dati[:,4], ls="-", label="$lay lay1 $pars", c=algo_color[lay])
    ax3.plot(dati[:,1], dati[:,5], ls="-", label="$lay lay1 $pars", c=algo_color[lay])
    #ax3.plot(dati[:,1], dati[:,5], label="qab (first layer)", color="orange")
    
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

    μ_train = mean(train_sgd) .* 100.
    σ_train = std(train_sgd) .* 100.
    μ_test = mean(test_sgd) .* 100.
    σ_test = std(test_sgd) .* 100.

    ax1.errorbar(epoche[1], μ_train, σ_train, ls="-", c=algo_color[:sgd],
                 capsize=0, label="train bin-sgd bs=$batchsize, lr=$lrsgd")
    ax1.errorbar(epoche[1], μ_test, σ_test, ls="--", ms=1, c=algo_color[:sgd], 
                 capsize=0, label="test bin-sgd bs=$batchsize, lr=$lrsgd")
end

ax1.set_xlabel("epochs", fontsize=12)
ax2.set_ylabel("q0", fontsize=10)#, color=colorb)
ax2.set_xlabel("epochs", fontsize=10)

ax2.tick_params(labelsize=7)

ax3.set_xlabel("epochs", fontsize=8)
ax3.set_ylabel("qab", fontsize=8)#, color=colorb)
ax3.tick_params(labelsize=7)

#ax2.set_ylabel("q0", fontsize=12, color=colorb)
#ax3.set_ylabel("qab", fontsize=12, color="orange")

ax2.set_ylim(0,1)
#ax3.set_ylim(0,1)

ax1.legend(loc="upper right", frameon=false, fontsize=9)
ax2.legend(loc="best", frameon=false, fontsize=8)
ax3.legend(loc="best", frameon=false, fontsize=8)

#plt.grid(false)

classt = multiclass ? "10class" : "2class"
Pstring = "$P"[1] * "e$(length("$(Int(P))")-1)"
dset_tit = dataset == :mnist ? "MNIST" :
           dataset == :fashion ? "FashionMNIST" :
           dataset == :cifar ? "CIFAR10" : "?"
fig.suptitle("$dset_tit $classt P=$(Pstring), bs=$batchsize, K=$(K[2:end-1]), ψ=$ψ, init=$ϵinit, iters=$maxiters, r=$r")
#fig.tight_layout()

#fig.savefig("deepMP_bs$(batchsize)_K$(K)_rho$(ρ1)_ψ_$(ψ)_P$(P)_maxiters_$(maxiters)_r$(r)_ϵinit_$(ϵinit)_.png")
fig.savefig("figure_deepMP.png")

plt.close()

# FIGURE 2
nlays = length(K)-1
fig, ax = plt.subplots(nlays,2)

for (i,(lay, ρ)) in enumerate(zip(lays, ρs))
        
    if !multiclass
        layers = [lay for i in 1:(length(K)-1)]
    else
        layers = [[lay for i in 1:(length(K)-2)]..., :argmax]
    end
    
    resfile = "../scripts/results/res_dataset$(dataset)_"
    resfile *= "Ks$(K)_bs$(batchsize)_layers$(layers)_rho$(ρ)_r$(r)_damp$(ψ)"
    resfile *= "_density$(density)"
    resfile *= "_M$(Int(P))_ϵinit$(ϵinit)_maxiters$(maxiters)"
    seed ≠ -1 && (resfile *= "_seed$(seed)")
    resfile *= ".dat"
    
    @show resfile

    dati = readdlm(resfile)

    ax[1].plot(dati[:,1], dati[:,4], ls="-", label="q0 lay1 $lay", c=algo_color[lay])
    ax[1+nlays].plot(dati[:,1], dati[:,5], ls="-", label="qab lay1 $lay", c=algo_color[lay])

end

ax[1].legend(loc="best", frameon=false, fontsize=10)
ax[1+nlays].legend(loc="best", frameon=false, fontsize=10)

fig.savefig("figure_deepMP2.png")

plt.close()
