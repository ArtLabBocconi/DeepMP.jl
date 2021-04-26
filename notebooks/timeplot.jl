using DelimitedFiles
using PyPlot

plt.style.use("default")
plt.style.use("seaborn-whitegrid")

rd(x, n) = round(x, sigdigits=n)

P = "6e4"
K = [28*28, 101, 101, 1]

lays = [:bp, :tap, :bpi]
batchsizes = [1, 10, 100, 1000]

ψ = 0.5
density = 1

fig, ax1 = plt.subplots(1)

algo_color = Dict(:sgd=>"black", :bp=>"tab:red", :tap=>"tab:green", :bpi=>"tab:blue")
algo_mark = Dict(:sgd=>"x", :bp=>"o", :tap=>"^", :bpi=>"s")

times_cpu = Dict(:sgd=>[60, 11, 2.3, 1.], 
             :bp=>[4000, 540, 360, 93], 
             :tap=>[2660, 376, 80, 6.5], 
             :bpi=>[3080, 455, 92, 36])

#times_gpu = Dict(:sgd=>[NaN, NaN, NaN, NaN], 
#             :bp=>[10, 10, 10, 10], 
#             :tap=>[100, 100, 100, 100], 
#             :bpi=>[1000, 1000, 1000, 1000])


for lay in [lays..., :sgd]
    device = lay == :sgd ? "GPU" : "CPU, 12 threads"
    ax1.plot(batchsizes, times_cpu[lay], marker=algo_mark[lay], ls=":", label="$lay ($device)", c=algo_color[lay])
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
plt.close()