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
Ks = [[0, 101, 101, 0], [0, 501, 501, 501, 0], [0, 501, 501, 501, 501, 501, 0], [0, 1024, 1024, 1024, 1024, 1024, 0]]
ρs = [[1.0, 1.0, 0.9], [1.0, 1.0, 1.0, 0.9], [1.0, 1.0, 1.0, 1.0, 1.0, 0.9], [1.0, 1.0, 1.0, 1.0, 1.0, 0.9]]
ψs = [[0.8, 0.8, 0.8], [0.8, 0.8, 0.8, 0.8], [0.8, 0.8, 0.8, 0.8, 0.8, 0.8], [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]]

figure_index = 3

lrsgd = 0.001
density = 1.0

# for different file names
lays = [:bpi, :tap, :mf]
lay_to_skip = [:tap, :mf]
lay_to_skip = []

multiclass = true
plot_sgd, plot_bp, plot_bayes = true, true, false
plot_overlaps = false
plot_adam = false

if multiclass
    seed_bp = [2, 7, 11]
    seed_sgd = [2, 7, 11]
    P = dataset ≠ :cifar10 ? 6e4 : 5e4
    maxiters = 1
    ϵinits = [1.0, 1.0, 1.0]

    K = Ks[figure_index]
    K[1] = Nin
    K[end] = 10
    L = length(K)-1
    r = [0.0 for _=1:L]
    ρ = ρs[figure_index]
    ψ = ψs[figure_index]

else

    seed_bp = [2, 7, 11]
    seed_sgd = [2, 7, 11]
    P = dataset ≠ :cifar10 ? 6e4 : 5e4
    maxiters = 1   
    r = [0.0 for _=1:length(K)-1]
    ϵinits = [1.0, 1.0, 1.0]

    ρs = [[1.0+1e-4, 1.0+1e-4, 0.9], [1.0+1e-4, 1.0+1e-4, 0.9], [1.0+1e-4, 1.0+1e-4, 0.9]]
    ψs = [[0.8, 0.8, 0.8], [0.81, 0.81, 0.81], [0.8, 0.8, 0.8]]

end

algo_color = Dict(:sgd=>"black", :bp=>"tab:red", :tap=>"tab:green", :bpi=>"tab:blue", :mf=>"tab:orange")
algo_mark = Dict(:sgd=>"o", :bp=>"^", :tap=>"s", :bpi=>"x", :mf=>"D")
errev = 10

# FIGURE 1
if plot_overlaps
    fig = plt.figure(constrained_layout=true, figsize=(6.4*1.7,4.8*1.1))
    gs = fig.add_gridspec(3, 4)
    ax1 = fig.add_subplot(py"$(gs)[:, 0:2]")
    ax2 = fig.add_subplot(py"$(gs)[0, 2]")
    ax3 = fig.add_subplot(py"$(gs)[0, 3]")
    ax4 = fig.add_subplot(py"$(gs)[1, 2]")
    ax5 = fig.add_subplot(py"$(gs)[1, 3]")
    ax6 = fig.add_subplot(py"$(gs)[2, 2]")
    ax7 = fig.add_subplot(py"$(gs)[2, 3]")
else
    fig = plt.figure(constrained_layout=true, figsize=(6.4,4.8))
    gs = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(py"$(gs)[:]")
end

if plot_bp 
    for (i,(lay, ρ, ψ, ϵinit)) in enumerate(zip(lays, [ρs[figure_index] for _=1:length(lays)], [ψs[figure_index] for _=1:length(lays)], ϵinits))
        
        lay in lay_to_skip && continue

        if !multiclass
            layers = [lay for i in 1:(length(K)-1)]
        else
            layers = [[lay for i in 1:(length(K)-2)]..., :argmax]
        end
        
        epoche_bp, train_bp, test_bp = [],[], []
        q0lay1, qablay1 = [], []
        q0lay2, qablay2 = [], []
        q0lay3, qablay3 = [], []

        train_bayes, test_bayes = [], []

        for seed in seed_bp
            resfile = "../scripts/resultsreb/res_dataset$(dataset)_"
            resfile *= "Ks$(K)_bs$(batchsize)_layers$(layers[1])_rho$(ρ)_r$(r)_damp$(ψ)"
            resfile *= "_density$(density)"
            resfile *= "_M$(Int(P))_ϵinit$(ϵinit)_maxiters$(maxiters)"
            seed ≠ -1 && (resfile *= "_seed$(seed)")
            resfile *= ".dat"

            if isfile(resfile) && filesize(resfile) ≠ 0
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

                if plot_bayes
                    push!(train_bayes, dati[:, end-1])
                    push!(test_bayes, dati[:, end])
                end

            else
                println("NOT FOUND: $resfile")
            end
        end

        μ_train_bp, σ_train_bp = mean(train_bp), std(train_bp)
        μ_test_bp, σ_test_bp = mean(test_bp), std(test_bp)

        if plot_bayes
            μ_train_bayes, σ_train_bayes = mean(train_bayes), std(train_bayes)
            μ_test_bayes, σ_test_bayes = mean(test_bayes), std(test_bayes)    
        end

        μ_q0lay1, σ_q0lay1 = mean(q0lay1), std(q0lay1)
        μ_qablay1, σ_qablay1 = mean(qablay1), std(qablay1)
        μ_q0lay2, σ_q0lay2 = mean(q0lay2), std(q0lay2)
        μ_qablay2, σ_qablay2 = mean(qablay2), std(qablay2)
        μ_q0lay3, σ_q0lay3 = mean(q0lay3), std(q0lay3)
        μ_qablay3, σ_qablay3 = mean(qablay3), std(qablay3)

        pars = "ρ=$([ρ[l] for l=1:L])"
        train_legend = "$(rd(μ_train_bp[end],3)) ± $(rd(σ_train_bp[end],3))"
        test_legend = "$(rd(μ_test_bp[end],3)) ± $(rd(σ_test_bp[end],3))"

        LAY = lay == :bp ? "BP" :
              lay == :bpi ? "BPI" :
              lay == :tap ? "AMP" :
              lay == :mf ? "MF" : error("unknown layer type")

        if plot_overlaps
            lbl_train = "$LAY train $pars, $train_legend"
            lbl_test = "$LAY test $pars, $test_legend"
        else
            lbl_train = "$LAY train $pars"
            lbl_test = "$LAY test $pars"
        end

        lbl_train = "$LAY train"
        lbl_test = "$LAY test"

        if plot_bayes
            ax1.plot(epoche_bp[1], μ_train_bayes, ls="-.", lw=2, label="Bayes "*lbl_train, color="tab:red", alpha=1.0)
            ax1.plot(epoche_bp[1], μ_test_bayes, ls=":", lw=2, label="Bayes "*lbl_test, color="tab:red", alpha=1.0)    
        
            ax1.fill_between(epoche_bp[1], μ_train_bayes-σ_train_bayes, μ_train_bayes+σ_train_bayes,
            color="tab:red", alpha=0.3, edgecolor=nothing)
ax1.fill_between(epoche_bp[1], μ_test_bayes-σ_test_bayes, μ_test_bayes+σ_test_bayes,
            color="tab:red", alpha=0.3, edgecolor=nothing)

        end
        ax1.plot(epoche_bp[1], μ_train_bp, ls="-", label=lbl_train, color=algo_color[lay])
        ax1.plot(epoche_bp[1], μ_test_bp, ls="--", label=lbl_test, color=algo_color[lay])

        ax1.fill_between(epoche_bp[1], μ_train_bp-σ_train_bp, μ_train_bp+σ_train_bp,
                        color=algo_color[lay], alpha=0.3, edgecolor=nothing)
        ax1.fill_between(epoche_bp[1], μ_test_bp-σ_test_bp, μ_test_bp+σ_test_bp,
                        color=algo_color[lay], alpha=0.3, edgecolor=nothing)

        if plot_overlaps

            ax2.plot(epoche_bp[1], μ_q0lay1, ls="-",
                label="$lay lay1", c=algo_color[lay])
            ax3.plot(epoche_bp[1], μ_qablay1, ls="-",
                label="$lay lay1", c=algo_color[lay])
            ax4.plot(epoche_bp[1], μ_q0lay2, ls="-",
                label="$lay lay2", c=algo_color[lay])
            ax5.plot(epoche_bp[1], μ_qablay2, ls="-",
                label="$lay lay2", c=algo_color[lay])
            ax6.plot(epoche_bp[1], μ_q0lay3, ls="-",
                label="$lay lay3", c=algo_color[lay])
            ax7.plot(epoche_bp[1], μ_qablay3, ls="-",
                label="$lay lay3", c=algo_color[lay])

            ax2.fill_between(epoche_bp[1], μ_q0lay1-σ_q0lay1, μ_q0lay1+σ_q0lay1,
                        color=algo_color[lay], alpha=0.3)
            ax3.fill_between(epoche_bp[1], μ_qablay1-σ_qablay1, μ_qablay1+σ_qablay1,
                        color=algo_color[lay], alpha=0.3)
            ax4.fill_between(epoche_bp[1], μ_q0lay2-σ_q0lay2, μ_q0lay2+σ_q0lay2,
                        color=algo_color[lay], alpha=0.3)
            ax5.fill_between(epoche_bp[1], μ_qablay2-σ_qablay2, μ_qablay2+σ_qablay2,
                        color=algo_color[lay], alpha=0.3)
            ax6.fill_between(epoche_bp[1], μ_q0lay3-σ_q0lay3, μ_q0lay3+σ_q0lay3,
                        color=algo_color[lay], alpha=0.3)
            ax7.fill_between(epoche_bp[1], μ_qablay3-σ_qablay3, μ_qablay3+σ_qablay3,
                        color=algo_color[lay], alpha=0.3)
        
        end

        println("$lay: train: $(rd(μ_train_bp[end],3)) ± $(rd(σ_train_bp[end],3)); test: $(rd(μ_test_bp[end],3)) ± $(rd(σ_test_bp[end],3))")

    end
end

Ksgd = K[2:end-1]
classes = multiclass ? nothing : []
dset_sgd = dataset==:cifar10 ? :cifar : dataset

if plot_sgd
    epoche, train_sgd, test_sgd = [], [], []
    for seedgd in seed_sgd
        file = "../../representations/knet/scripts/resultsreb/res_dataset$(dset_sgd)_classes$(classes)_binwtrue_hidden$(Ksgd)_biasfalse_freezetopfalse"
        (P > 0 && (P≠6e4) && P≠5e4) && (file *= "_P$(Int(P))")
        file *= "_lr$(lrsgd)_bs$(batchsize)"
        seedgd ≠ 2 && (file *= "_seed$(seedgd)")
        file *= ".dat"
        @show file

        if isfile(file)
            dati_sgd = readdlm(file)
            push!(epoche, dati_sgd[:, 1])
            push!(train_sgd, dati_sgd[:, 2])
            push!(test_sgd, dati_sgd[:, 3])
        else
            println("* NOT FOUND: $file")
        end
    end

    μ_train, σ_train = mean(train_sgd) .* 100., std(train_sgd) .* 100.
    μ_test, σ_test = mean(test_sgd) .* 100., std(test_sgd) .* 100.

    #μ_test .+= 0.3

    train_legend = "$(rd(μ_train[end],2)) ± $(rd(σ_train[end],2))"
    test_legend = "$(rd(μ_test[end],2)) ± $(rd(σ_test[end],2))"

    if plot_overlaps
        lbl_train = "binaryNet train bs=$batchsize, lr=$lrsgd, $train_legend"
        lbl_test = "binaryNet test bs=$batchsize, lr=$lrsgd, $test_legend"
    else
        lbl_train = "binaryNet train, lr=$lrsgd"
        lbl_test = "binaryNet test, lr=$lrsgd"
    end

    lbl_train = "BinaryNet train"
    lbl_test = "BinaryNet test"

    ax1.plot(epoche[1], μ_train, ls="-", c=algo_color[:sgd], label=lbl_train, alpha=1.0)
    ax1.plot(epoche[1], μ_test, ls="--", c=algo_color[:sgd], label=lbl_test, alpha=1.0)

    ax1.fill_between(epoche[1], μ_train+σ_train, μ_train-σ_train, color=algo_color[:sgd], alpha=0.3)
    ax1.fill_between(epoche[1], μ_test+σ_test, μ_test-σ_test, color=algo_color[:sgd], alpha=0.3)

    println("SGD: train: $(rd(μ_train[end],2)) ± $(rd(σ_train[end],2)); test: $(rd(μ_test[end],2)) ± $(rd(σ_test[end],2))")

end

if plot_adam
    epoche, train_sgd, test_sgd = [], [], []
    for seedgd in seed_sgd
        file = "../../representations/knet/scripts/resultsreb/res_dataset$(dset_sgd)_classes$(classes)_binwtrue_hidden$(Ksgd)_biasfalse_freezetopfalse"
        (P > 0 && (P≠6e4) && P≠5e4) && (file *= "_P$(Int(P))")
        file *= "_lr$(lrsgd)_bs$(batchsize)"
        seedgd ≠ 2 && (file *= "_seed$(seedgd)")
        file *= ".dat"
        @show file
        
        if isfile(file)
            dati_sgd = readdlm(file)
            push!(epoche, dati_sgd[:, 1])
            push!(train_sgd, dati_sgd[:, 2])
            push!(test_sgd, dati_sgd[:, 3])
        else
            println("* NOT FOUND: $file")
        end
    end

    μ_train, σ_train = mean(train_sgd) .* 100., std(train_sgd) .* 100.
    μ_test, σ_test = mean(test_sgd) .* 100., std(test_sgd) .* 100.

    #μ_test .+= 0.3

    train_legend = "$(rd(μ_train[end],2)) ± $(rd(σ_train[end],2))"
    test_legend = "$(rd(μ_test[end],2)) ± $(rd(σ_test[end],2))"

    if plot_overlaps
        lbl_train = "binaryNet train bs=$batchsize, lr=$lrsgd, $train_legend"
        lbl_test = "binaryNet test bs=$batchsize, lr=$lrsgd, $test_legend"
    else
        lbl_train = "binaryNet train, lr=$lrsgd"
        lbl_test = "binaryNet test, lr=$lrsgd"
    end

    lbl_train = "BinaryNet train"
    lbl_test = "BinaryNet test"

    ax1.plot(epoche[1], μ_train, ls="-", c=algo_color[:sgd], label=lbl_train, alpha=1.0)
    ax1.plot(epoche[1], μ_test, ls="--", c=algo_color[:sgd], label=lbl_test, alpha=1.0)

    ax1.fill_between(epoche[1], μ_train+σ_train, μ_train-σ_train, color=algo_color[:sgd], alpha=0.3)
    ax1.fill_between(epoche[1], μ_test+σ_test, μ_test-σ_test, color=algo_color[:sgd], alpha=0.3)

    println("SGD: train: $(rd(μ_train[end],2)) ± $(rd(σ_train[end],2)); test: $(rd(μ_test[end],2)) ± $(rd(σ_test[end],2))")

end


if dataset == :mnist
    if multiclass
        ax1.set_ylim(0, 50)
    else
        ax1.set_ylim(0, 5)
    end
elseif dataset == :fashion
    if multiclass
        ax1.set_ylim(0, 25)
    else
        ax1.set_ylim(0, 8)
    end
elseif dataset == :cifar10
    if multiclass
        ax1.set_ylim(0, 90)
    else
        ax1.set_ylim(0, 50)
    end
end

ax1.set_xlabel("epochs", fontsize=18)
ax1.set_ylabel("error (%)", fontsize=18)
ax1.tick_params(labelsize=14)
ax1.legend(loc="upper right", frameon=false, fontsize=14, ncol=2)

if plot_overlaps
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
end

#ax3.tick_params(labelsize=7)
#ax2.set_ylim(0,1)

if plot_overlaps
    ax2.legend(loc="best", frameon=false, fontsize=10)
    ax3.legend(loc="best", frameon=false, fontsize=10)
    ax4.legend(loc="best", frameon=false, fontsize=10)
    ax5.legend(loc="best", frameon=false, fontsize=10)
    ax6.legend(loc="best", frameon=false, fontsize=10)
    ax7.legend(loc="best", frameon=false, fontsize=10)
end

#plt.grid(false)

#plt.subplots_adjust(hspace=0.2, wspace=0.4)

classt = multiclass ? "10class" : "2class"
Pstring = "$P"[1] * "e$(length("$(Int(P))")-1)"
dset_tit = dataset == :mnist ? "MNIST" :
           dataset == :fashion ? "FashionMNIST" :
           dataset == :cifar10 ? "CIFAR10" : "?"
#fig.suptitle("$dset_tit $classt P=$(Pstring), bs=$batchsize, K=$(K[2:end-1]), ψ=$(ψs[end]), init=$(ϵinits[1]), iters=$maxiters, r=$r")
fig.suptitle("$dset_tit $classt P=$(Pstring), density=$(density*100)% bs=$batchsize, K=$(K[2:end-1]) iters=$maxiters", fontsize=14)

fig.tight_layout()

#fig.savefig("figures/deepMP_bs$(batchsize)_K$(K)_rho$(ρ1)_ψ_$(ψ)_P$(P)_maxiters_$(maxiters)_r$(r)_ϵinit_$(ϵinit)_.png")
fig.savefig("figures/figure_1.png")
multc = multiclass ? "multiclass" : "2class"
#fig.savefig("figures/figBP_$(K[2:end-1]).$(dataset).$(multc)_sparse.png")
ovs = plot_overlaps ? ".ovs" : ""
bay = plot_bayes ? ".bayes" : ""
#fig.savefig("figures/figBP_$(K[2:end-1]).$(dataset).$(multc)$(ovs)$(bay)_sparse.pdf")

plt.close()