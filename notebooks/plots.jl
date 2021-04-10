using DelimitedFiles
using PyPlot

K = [28*28, 101, 101, 1]
batchsize = 250

K = [28*28, 101, 101, 101, 1]
batchsize = 1000

K = [28*28, 101, 1]
batchsize = 10

ρ = 1.0001
r = 0.
density = 1
layers = [:bp for i in 1:(length(K)-1)]

resfile = "../results/res_Ks$(K)_bs$(batchsize)_layers$(layers)_rho$(ρ)_r$(r)_density$(density).dat"
dati = readdlm(resfile)

fig, (ax1, ax2) = plt.subplots(2,1)

ax1.plot(dati[:,1], dati[:,2], label="train error")
ax1.plot(dati[:,1], dati[:,3], label="test error")

#ax1.set_xlabel("epochs", fontsize=12)
ax1.set_ylabel("errors", fontsize=12)
ax1.set_ylim(0,15)

colorb = "tab:blue"
ax2.plot(dati[:,1], dati[:,4], label="q0", color=colorb)
ax2.plot(dati[:,1], dati[:,5], label="qab (first layer)", color="orange")
#ax3 = ax2.twinx()
#ax3.plot(dati[:,1], dati[:,5], label="qab (first layer)", color="orange")

ax2.set_xlabel("epochs", fontsize=12)
ax2.set_ylabel("overlap", fontsize=12)#, color=colorb)
#ax2.set_ylabel("q0", fontsize=12, color=colorb)
#ax3.set_ylabel("qab", fontsize=12, color="orange")

ax2.set_ylim(0,1)
ax3.set_ylim(0,1)

ax1.legend()
ax2.legend()
#ax2.legend(loc=(0.1, 0.8))
#ax3.legend(loc=(0.1, 0.6))

plt.suptitle("MNIST even vs odd, P=1e4, K=$K, bs=$batchsize, ρ = $ρ")
plt.tight_layout()
plt.savefig("deepMP_2layers.pdf")
plt.close()