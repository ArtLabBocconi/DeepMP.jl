using DelimitedFiles, Statistics
using PyPlot
using PyCall
using Printf

plt.style.use("default")
plt.style.use("seaborn-whitegrid")
cd("/home/fabrizio/workspace/DeepMP.jl/notebooks")

dataset = "fashion"
Ksgd = [501, 501]

locen_sgd = readdlm("../scripts/results/localenergy_dataset$(dataset)_K$(Ksgd)_expsgd.dat")
locen_bp = readdlm("../scripts/results/localenergy_dataset$(dataset)_K$(Ksgd)_expbp.dat")

fig, ax1 = plt.subplots(1)

ax1.errorbar(locen_sgd[:,1], locen_sgd[:,2] .- locen_sgd[1,2], locen_sgd[:,3], ls="-", errorevery=1,
label="BinaryNet", marker="v", c="black")
ax1.errorbar(locen_sgd[:,1], locen_bp[:,2] .- locen_bp[1,2], locen_bp[:,3], ls="-", errorevery=1,
label="BPI", marker="s", c="tab:blue")

ax1.set_xlabel("flip probability p", fontsize=16)
ax1.set_ylabel("local energy", fontsize=16)
ax1.tick_params(labelsize=14)

ax1.legend(loc="upper left", frameon=false, fontsize=14)

fig.tight_layout()
fig.savefig("figures/local_energy.png")
fig.savefig("figures/local_energy.pdf")