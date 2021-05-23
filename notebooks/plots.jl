using DelimitedFiles
using PyPlot

plt.style.use("default")
plt.style.use("seaborn-whitegrid")

rd(x, n) = round(x, sigdigits=n)

# for different file names
saveres = true
lays = [:bp, :bpi, :tap]
lays = [:bp, :bpi]

final_params = false
if !final_params
    batchsize = 1   # 1, 16, 128, 1024
    ρ1 = 1e-6       # -1e-1, -1e-5, 0., 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1
    ψ = 0.8         # ψs = [0:0.2:1;], [0.9, 0.99, 0.999, 0.9999]
    P = 6e4         # 1e2, 1e3, 1e4, 6e4 (bs = 1e0, 1e1, 1e2, 6e2 respectively)
    maxiters = 1    # 1, 10, 50, 100
    r = 0.          # [0:0.2:1.2;] (for maxiters=10)
    ϵinit = 1e0     # 0., 1e-3, 1e-2, 1e-1, 5e-1, 1e0
    K = [28*28, 501, 501, 501, 1] # [[28*28, 1-5-10-01, (1-5-10-01), (1-5-10-01), 1]]
    ρs = [ρ1, ρ1, ρ1] .+ 1.
else
    # for beautiful final figure
    batchsize = 128
    ρ1 = 1e-5       
    ψ = 0.8         
    P = 6e4         
    maxiters = 1   
    r = 0.        
    ϵinit = 1e0
    K = [28*28, 101, 101, 1]
    ρs = [ρ1, ρ1*10, ρ1] .+ 1.
end

density = 1

fig, ax1 = plt.subplots(1)
ax2 = ax1.inset_axes([0.27, 0.575, 0.35, 0.4])
ax3 = ax1.inset_axes([0.525, 0.2, 0.35, 0.275])

algo_color = Dict(:sgd=>"black", :bp=>"tab:red", :tap=>"tab:green", :bpi=>"tab:blue")
algo_mark = Dict(:sgd=>"o", :bp=>"^", :tap=>"s", :bpi=>"x")

for (i,(lay, ρ)) in enumerate(zip(lays, ρs))
        
    layers = [lay for i in 1:(length(K)-1)]
    
    resfile = "../scripts/results/res_Ks$(K)_bs$(batchsize)_layers$(layers)_rho$(ρ)_r$(r)_damp$(ψ)_density$(density).dat"
    if saveres
        resfile = "../scripts/results/res_"
        resfile *= "Ks$(K)_bs$(batchsize)_layers$(layers)_rho$(ρ)_r$(r)_damp$(ψ)"
        resfile *= "_density$(density)"
        resfile *= "_M$(Int(P))_ϵinit$(ϵinit)_maxiters$(maxiters)"
        resfile *= ".dat"
    end
    
    @show resfile

    dati = readdlm(resfile)

    pars = "ρ=$(rd(ρ-1,1))"

    ax1.plot(dati[:,1], dati[:,2], ls="-", label="train $lay $pars", c=algo_color[lay])
    ax1.plot(dati[:,1], dati[:,3], ls="--", label="test $lay $pars", c=algo_color[lay])

    #ax1.set_xlabel("epochs", fontsize=12)
    ax1.set_ylabel("error (%)", fontsize=12)
    ax1.set_ylim(0,30)

    ax2.plot(dati[:,1], dati[:,4], ls="-", label="$lay $pars", c=algo_color[lay])
    ax3.plot(dati[:,1], dati[:,5], ls="-", label="$lay lay1 $pars", c=algo_color[lay])
    #ax3.plot(dati[:,1], dati[:,5], label="qab (first layer)", color="orange")
    
end

Ksgd = K[2:end-1]
file = "../../representations/knet/results/res_datasetfashion_classesAny[]_binwtrue_hidden$(Ksgd)_biasfalse_freezetopfalse_lr1.0_bs$(batchsize).dat"

dati_sgd = readdlm(file)

ax1.plot(dati_sgd[:,1], dati_sgd[:,2].*100., ls="-", label="train bin-sgd bs=$batchsize", c=algo_color[:sgd])
ax1.plot(dati_sgd[:,1], dati_sgd[:,3].*100., ls="--", ms=1, label="test bin-sgd bs=$batchsize", c=algo_color[:sgd])

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

fig.suptitle("MNIST even vs odd, P=$(Int(P)), K=$K, bs=$batchsize, ψ=$ψ")
#fig.tight_layout()
#fig.savefig("deepMP_bs$(batchsize)_K$(K)_comparison.png")
fig.savefig("figure_deepMP.png")

plt.close()



#if batchsize == 1000
#    ρs = [1.00001, 1.00001, 1.00001]
#elseif batchsize == 100
#    ρs = [1.00001, 1.00001, 1.00001]
#elseif batchsize == 10
#    ρs = [1.00001, 1.000001, 1.00001]
#elseif batchsize == 1
#    ρs = [1.000001, 1.000001, 1.000001]
#end