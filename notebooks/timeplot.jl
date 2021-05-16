using DelimitedFiles
using PyPlot

plt.style.use("default")
plt.style.use("seaborn-whitegrid")

rd(x, n) = round(x, sigdigits=n)

P = "6e4"
K = [28*28, 101, 101, 1]

lays = [:bp, :tap, :bpi]
batchsizes = [1, 16, 128, 1024]

ψ = 0.5
density = 1

fig, ax1 = plt.subplots(1)

algo_color = Dict(:sgd=>"black", :bp=>"tab:red", :tap=>"tab:green", :bpi=>"tab:blue")
algo_mark = Dict(:sgd=>"x", :bp=>"o", :tap=>"^", :bpi=>"s")

times_cpu = Dict(:sgd=>[600, 107, 23, 11], 
             :bp=>[4000, 540, 360, 93], 
             :tap=>[3350, 376, 80, 11], 
             :bpi=>[3700, 455, 92, 36])

times_gpu = Dict(:sgd=>[48, 3, 0.45, 0.2], 
             :bp=>[118, 7.5, 2.5, 2.5], 
             :tap=>[104, 6.5, 0.8, 0.3], 
             :bpi=>[91, 6.3, 1.2, 0.8])

# almeno per SGD mi sa che c'è un memory leakage (non ha senso che bs minore occupa di più oltretutto)
memory_gpu = Dict(:sgd=>[6e3, 4e3, 4e3, NaN], 
             :bp=>[15e3, 22.5e3, 21e3, 22e3], 
             :tap=>[8e3, 8e3, 8e3, 22e3], 
             :bpi=>[14e3, 22e3, 21e3, 21e3])


for lay in [lays..., :sgd]
    device = lay == :sgd ? "GPU" : "GPU"
    ax1.plot(batchsizes, times_gpu[lay], marker=algo_mark[lay], ls=":", label="$lay ($device)", c=algo_color[lay])
    #ax1.plot(batchsizes, times_gpu[lay], ls="-", label="$lay", c=algo_color[lay])
end

ax1.set_xlabel("batch size", fontsize=12)
ax1.set_ylabel("time per epoch (s)", fontsize=12)
ax1.tick_params(labelsize=12)

ax1.set_xscale("log")
ax1.set_yscale("log")

ax1.legend(loc="best", frameon=false, fontsize=12)

fig.suptitle("MNIST even vs odd, P=$P, K=$K")
#fig.tight_layout()
fig.savefig("deepMP_times_K$(K).pdf")
#plt.close()