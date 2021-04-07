using DelimitedFiles
using PyPlot

K = [28*28, 101, 101, 1]
batchsize = 250
layers = [:bp, :bp, :bp]

K = [28*28, 101, 101, 101, 1]
batchsize = 1000
layers = [:bp, :bp, :bp, :bp, :bp]

ρ = 1.
r = 0.
density = 1

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
ax3 = ax2.twinx()
ax3.plot(dati[:,1], dati[:,5], label="qab (first layer)", color="orange")

ax2.set_xlabel("epochs", fontsize=12)
ax2.set_ylabel("q0", fontsize=12, color=colorb)
ax3.set_ylabel("qab", fontsize=12, color="orange")

ax1.legend()
ax2.legend(loc=(0.6, 0.6))
ax3.legend(loc=(0.6, 0.4))

plt.suptitle("MNIST even vs odd, P=60000, K=$K, bs=$batchsize")
plt.tight_layout()
plt.savefig("deepMP_2layers.pdf")
plt.close()