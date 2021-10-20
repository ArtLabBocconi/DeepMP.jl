using DelimitedFiles
using PyPlot

plt.style.use("default")
plt.style.use("seaborn-whitegrid")

rd(x, n) = round(x, sigdigits=n)

P = "6e4"
K = [28*28, 101, 101, 1]

lays = [:bp, :bpi, :tap, :mf]
batchsizes = [1, 16, 128, 1024]

ψ = 0.5
density = 1

fig, ax1 = plt.subplots(1)

algo_color = Dict(:sgd=>"black", :bp=>"tab:red", :tap=>"tab:green", :bpi=>"tab:blue", :mf=>"tab:orange")
algo_mark = Dict(:sgd=>"x", :bp=>"o", :tap=>"^", :bpi=>"s", :mf=>"v")

#for seed in seed_bp
#    resfile = "../scripts/results/res_dataset$(dataset)_"
#    resfile *= "Ks$(K)_bs$(batchsize)_layers$(layers)_rho$(ρ)_r$(r)_damp$(ψ)"
#    resfile *= "_density$(density)"
#    resfile *= "_M$(Int(P))_ϵinit$(ϵinit)_maxiters$(maxiters)"
#    seed ≠ -1 && (resfile *= "_seed$(seed)")
#    resfile *= ".dat"
#    @show resfile
#
#    dati = readdlm(resfile)
#
#    push!(tempi_bp, dati[:, end])
#end


times_gpu = Dict(:sgd=>[48, 3, 0.45, 0.2], 
             :bp=>[118, 7.5, 2.5, 2.5],  
             :bpi=>[91, 6.3, 1.2, 0.8],
             :tap=>[104, 6.5, 0.8, 0.3],
             :mf=>[96, 6.1, 0.7, 0.2])

for lay in [lays..., :sgd]
    device = lay == :sgd ? "(GPU)" : "(GPU)"
    device = ""
    LAY = lay == :bp ? "BP" :
          lay == :bpi ? "BPI" :
          lay == :tap ? "AMP" :
          lay == :mf ? "MF" : 
          lay == :sgd ? "BinaryNet" : error()
    ax1.plot(batchsizes, times_gpu[lay], marker=algo_mark[lay], ls=":", label="$LAY $device", c=algo_color[lay])
    #ax1.plot(batchsizes, times_gpu[lay], ls="-", label="$lay", c=algo_color[lay])
end

ax1.set_xlabel("batch size", fontsize=18)
ax1.set_ylabel("time per epoch (s)", fontsize=18)
ax1.tick_params(labelsize=16)

ax1.set_xscale("log")
ax1.set_yscale("log")

ax1.legend(loc="best", frameon=false, fontsize=16)

#fig.suptitle("FashionMNIST 2 classes, P=$P, K=$K")
fig.tight_layout()
fig.savefig("figBP_times_K$(K[2:end-1]).fashion.2class.png")
fig.savefig("figBP_times_K$(K[2:end-1]).fashion.2class.pdf")
#plt.close()


## almeno per SGD mi sa che c'è un memory leakage (non ha senso che bs minore occupa di più oltretutto)
#memory_gpu = Dict(:sgd=>[6e3, 4e3, 4e3, NaN], 
#             :bp=>[15e3, 22.5e3, 21e3, 22e3], 
#             :tap=>[8e3, 8e3, 8e3, 22e3], 
#             :bpi=>[14e3, 22e3, 21e3, 21e3])
#
#times_cpu = Dict(:sgd=>[600, 107, 23, 11], 
#             :bp=>[4000, 540, 360, 93], 
#             :tap=>[3350, 376, 80, 11], 
#             :bpi=>[3700, 455, 92, 36])
