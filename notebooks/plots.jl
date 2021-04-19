using DelimitedFiles
using PyPlot

plt.style.use("default")
plt.style.use("seaborn-deep")

P = "6e4"

K = [28*28, 101, 101, 1]
batchsize = 250

K = [28*28, 101, 101, 101, 1]
batchsize = 1000

K = [28*28, 101, 101, 1]
batchsize = 1

if batchsize == 10
    ρs = [1.0001, 1.00001]
else
    ρs = [1.00001, 1.000001]
end

r = 0.
density = 1
layers = [:bp for i in 1:(length(K)-1)]

fig, ax1 = plt.subplots(1)
ax2 = ax1.inset_axes([0.27, 0.5, 0.35, 0.4])
ax3 = ax1.inset_axes([0.75, 0.4, 0.23, 0.25])

for (i,ρ) in enumerate(ρs)
    resfile = "../results/res_Ks$(K)_bs$(batchsize)_layers$(layers)_rho$(ρ)_r$(r)_density$(density).dat"
    dati = readdlm(resfile)

    pars = "ρ=$ρ"
    
    ls = i==1 ? ":" :
         i==2 ? "-" : ""
    
    ax1.plot(dati[:,1], dati[:,2], ls=ls, label="train bp $pars")
    ax1.plot(dati[:,1], dati[:,3], ls=ls, label="test bp $pars")

    #ax1.set_xlabel("epochs", fontsize=12)
    ax1.set_ylabel("error (%)", fontsize=12)
    ax1.set_ylim(0,30)

    #colorb = "tab:blue"
    ax2.plot(dati[:,1], dati[:,4], label="q0 $pars")#, color=colorb)
    ax3.plot(dati[:,1], dati[:,5], label="qab lay1 $pars")#, color="orange")
    #ax3.plot(dati[:,1], dati[:,5], label="qab (first layer)", color="orange")
end

dati_sgd = readdlm("../../representations/knet/results/res_datasetfashion_classesAny[]_binwtrue_hidden[101, 101]_biasfalse_freezetopfalse_lr1.0_bs1.dat")

ax1.plot(dati_sgd[:,1], dati_sgd[:,2].*100., ls="--", label="train bin-sgd")
ax1.plot(dati_sgd[:,1], dati_sgd[:,3].*100., ls="--", ms=1, label="test bin-sgd")

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

ax1.legend(loc="best", frameon=false, fontsize=9)
ax2.legend(loc="best", frameon=false, fontsize=8)
ax3.legend(loc="best", frameon=false, fontsize=5)

#plt.grid(false)

fig.suptitle("MNIST even vs odd, P=$P, K=$K, bs=$batchsize")
#fig.tight_layout()
fig.savefig("deepMP_2layers.pdf")
plt.close()