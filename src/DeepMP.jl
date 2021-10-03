module DeepMP

using ExtractMacro
using SpecialFunctions
using Printf
using Random
using LinearAlgebra
using Statistics
using Base: @propagate_inbounds # for DataLoader
using Tullio
using LoopVectorization
using CUDA, KernelAbstractions, CUDAKernels
using Adapt
using Functors
using JLD2
import Zygote
import ForwardDiff
import Cassette
CUDA.allowscalar(false)

# using PyPlot

const F = Float64
const CVec = Vector{Complex{F}}
const IVec = Vector{Int}
const Vec = Vector{F}
const VecVec = Vector{Vec}
const IVecVec = Vector{IVec}
const VecVecVec = Vector{VecVec}
const IVecVecVec = Vector{IVecVec}

include("cuda.jl")
include("utils/utils.jl")
include("utils/functions.jl")
include("utils/dataloader.jl")
include("utils/Magnetizations.jl"); using .Magnetizations
include("layers/layers.jl")
include("factor_graph.jl")
include("reinforcement.jl")

function converge!(g::FactorGraph;  maxiters=10000, ϵ=1f-5,
                                    altsolv=false,
                                    altconv=false,
                                    plotinfo=0,
                                    teacher=nothing,
                                    reinfpar,
                                    verbose=1,
                                    xtest=nothing,
                                    ytest=nothing)

    for it = 1:maxiters

        t = @timed Δ = update!(g, reinfpar)
        #E = energy(g)
        E = mean(vec(forward(g, g.layers[1].x)) .!= g.layers[end].y) * 100

        verbose >= 1 && @printf("it=%d \t (r=%f) Etrain=%.2f%% \t Δ=%f \n",
                                it, reinfpar.r, E, Δ)
        if verbose >= 2
            Etest = 100.0
            if ytest !== nothing
                Etest = mean(vec(forward(g, xtest)) .!= ytest) * 100
            end
            @printf("          Etest=%.2f%%  rstep=%g  t=%g\n", Etest, reinfpar.rstep, t.time)
        end

        plotinfo > 0 && plot_info(g, plotinfo, verbose=verbose)
        update_reinforcement!(reinfpar)
        if altsolv && E == 0
            verbose > 0 && println("Found Solution: correctly classified $(g.M) patterns.")
            return it, E, Δ
        end
        if altconv && Δ < ϵ
            verbose > 0 && println("Converged!")
            return it, E, Δ
        end
    end
    return maxiters, 1, 1.0
end

function solve(; K::Vector{Int} = [101, 3],
                 Kteacher = K,
                 α = 100., # num_examples/num_params. Ignore if M is used.
                 M = -1, # num_params. If negative, set from alpha.
                 Mtest = 10000, # num test samples
                 seedx::Int = -1,
                 density = 1,
                 TS = false,
                 hidden_manifold = false,
                 density_teacher = density,
                 kws...)

    seedx > 0 && Random.seed!(seedx)

    L = length(K) - 1
    density = process_density(density, L)
    numW = length(K)==2 ? K[1]*K[2]*density[1]  :
            sum(l->density[l] * K[l]*K[l+1], 1:length(K)-2)
    numW = round(Int, numW)

    if M <= 0
        M = round(Int, α * numW)
        α = M / numW
    end

    N = K[1]
    D = Kteacher[1]
    @assert hidden_manifold || N == D
    xtrain = rand(F[-1, 1], D, M)

    if TS
        teacher = rand_teacher(Kteacher; density=density_teacher)
        ytrain = Int.(forward(teacher, xtrain) |> vec)
        xtest = rand(F[-1, 1], D, Mtest)
        ytest = Int.(forward(teacher, xtest) |> vec)
    else
        teacher = nothing
        ytrain = rand([-1,1], M)
        xtest, ytest = nothing, nothing
    end

    if hidden_manifold
        features = rand(F[-1, 1], N, D)
        xtrain = sign.(features * xtrain ./ sqrt(D))
        if xtest !== nothing
            xtest = sign.(features * xtest ./ sqrt(D))
        end
    end

    @assert size(xtrain) == (N, M)
    @assert size(ytrain) == (M,)
    @assert all(x -> x == -1 || x == 1, ytrain)

    solve(xtrain, ytrain; K, density, teacher, xtest, ytest, kws...)
end


function solve(xtrain::AbstractMatrix, ytrain::AbstractVector;
                xtest = nothing, ytest = nothing,
                dataset = :fashion,
                K::Vector{Int},                # List of widths for each layer, e.g. [28*28, 101, 101, 1]
                layers,                        # List of layer types  e.g. [:bpi, :bpi, :argmax],
                maxiters = 100,
                ϵ = 1e-4,                      # convergence criterion
                r = 0., rstep = 0.,            # reinforcement parameters for W vars
                ψ = 0.,                        # damping coefficient
                yy = -1.,                      # focusing BP parameter
                h0 = nothing,                  # external field
                g0 = nothing,                  # factor graph init
                ρ = 1.,                        # coefficient for external field from mini-batch posterior
                rbatch = 0.,                   # reinforcement parameter for external field
                freezetop = false,             # freeze top-layer's weights to 1
                teacher = nothing,
                altsolv::Bool = true,
                altconv::Bool = false,
                seed::Int = -1,
                β = Inf,
                density = 1f0,                  # density of fully connected layer
                batchsize = -1,                 # only supported by some algorithms
                epochs = 100,
                ϵinit = 0.,
                plotinfo = 0,
                verbose = 1,
                usecuda = true,
                gpu_id = -1,
                saveres = false,
                )

    usecuda = CUDA.functional() && usecuda
    device =  usecuda ? gpu : cpu
    usecuda && gpu_id >= 0 && device!(gpu_id)
    if seed > 0
        Random.seed!(seed)
        usecuda && CUDA.seed!(seed)
    end

    L = length(K) - 1
    ψ = num_to_vec(ψ, L)
    ρ = num_to_vec(ρ, L)

    xtrain, ytrain = device(xtrain), device(ytrain)
    xtest, ytest = device(xtest), device(ytest)
    dtrain = DataLoader((xtrain, ytrain); batchsize, shuffle=true, partial=false)

    # g = FactorGraph(first(dtrain)..., K, ϵinit, layers; β, density, device)
    g0 !== nothing && @assert isa(g0, FactorGraph)
    if isa(g0, FactorGraph)
        g = deepcopy(g0)
    else
        g = FactorGraph(first(dtrain)..., K, ϵinit, layers; β, density, device)
    end
    h0 !== nothing && set_external_fields!(g, h0; ρ, rbatch);
    if teacher !== nothing
        teacher = device.(teacher)
        has_same_size(g, teacher) && set_weight_mask!(g, teacher)
    end
    # initrand!(g)
    g0 !== nothing || initrand!(g)
    freezetop && freezetop!(g, 1)
    reinfpar = ReinfParams(r, rstep, yy, ψ)

    if saveres
        resfile = "results/res_dataset$(dataset)_"
        resfile *= "Ks$(K)_bs$(batchsize)_layers$(layers)_rho$(ρ)_r$(r)_damp$(ψ)"
        resfile *= "_density$(density)"
        resfile *= "_M$(length(ytrain))_ϵinit$(ϵinit)_maxiters$(maxiters)"
        seed ≠ -1 && (resfile *= "_seed$(seed)")
        resfile *= ".dat"
        fres = open(resfile, "w")
    end

    function report(epoch; t=(@timed 0), converged=0., solved=0., meaniters=0.)
        Etrain = mean(vec(forward(g, xtrain)) .!= ytrain) * 100
        Etrain_bayes = bayesian_error(g, xtrain, ytrain) *100
        num_batches = length(dtrain)
        Etest = 100.0
        if ytest !== nothing
            Etest = mean(vec(forward(g, xtest)) .!= ytest) * 100
            Etest_bayes = bayesian_error(g, xtest, ytest) *100
        end

        verbose >= 1 && @printf("Epoch %i (conv=%g, solv=%g <it>=%g): Etrain=%.2f%% Etest=%.2f%%  r=%g rstep=%g ρ=%s  t=%g (layers=%s, bs=%d)\n",
                                epoch, (converged/num_batches), (solved/num_batches), (meaniters/num_batches),
                                Etrain, Etest, reinfpar.r, reinfpar.rstep, ρ, t.time, "$layers", batchsize)

        verbose >= 1 && @printf("\t\t\tEtrainBayes=%.2f%% EtestBayes=%.2f%%\n", Etrain_bayes, Etest_bayes)

        q0s, qWαβs = plot_info(g, 0; verbose)

        if saveres
            outf = @sprintf("%d %g %g", epoch, Etrain, Etest)
            for (q0, qWαβ) in zip(q0s, qWαβs)
                outf *= @sprintf(" %g %g", mean(q0), mean(qWαβ))
            end
            outf *= @sprintf(" %g", t.time)
            outf *= @sprintf(" %g %g", Etrain_bayes, Etest_bayes)
            println(fres, outf); flush(fres)
        end
        return Etrain
    end


    if batchsize <= 0
        ## FULL BATCH message passing
        it, e, δ = converge!(g; maxiters, ϵ, reinfpar,
                            altsolv, altconv, plotinfo,
                            teacher, verbose,
                            xtest, ytest)

    else
        ## MINI_BATCH message passing
        # TODO check reinfparams updates in mini-batch case
        report(0)
        for epoch = 1:epochs
            converged = solved = meaniters = 0
            t = @timed for (b, (x, y)) in enumerate(dtrain)
                all(x->x==0, ρ) || set_Hext_from_H!(g, ρ, rbatch)
                set_input_output!(g, x, y)

                it, e, δ = converge!(g; maxiters, ϵ,
                                        reinfpar, altsolv, altconv, plotinfo=0,
                                        teacher, verbose=verbose-1)
                converged += (δ < ϵ)
                solved    += (e == 0)
                meaniters += it

                verbose >= 2 && print("b = $b / $(length(dtrain))\r")

                #if epochs == 1 && b % 1000 == 0
                #    Etrain = report(b; converged, solved, meaniters)
                #end

            end
            Etrain = report(epoch; t, converged, solved, meaniters)
            #Etrain == 0 && break
        end
    end
    if saveres
        close(fres)
        println("outfile: $resfile")
        conf_file = "results/conf$(resfile[12:end-4]).jld2"
        @show conf_file
        #save(conf_file, Dict("weights" => getW(g)))
    end

    Etrain = sum(vec(forward(g, xtrain)) .!= ytrain)
    # return g, getW(g), teacher, Etrain, it, conf_file
    return g, getW(g), teacher, Etrain, it
end

function solveCL(xtrain::AbstractMatrix, ytrain::AbstractVector;
                num_tasks = 6,
                xtest = nothing, ytest = nothing,
                dataset = :fashion,
                K::Vector{Int},                # List of widths for each layer, e.g. [28*28, 101, 101, 1]
                layers,                        # List of layer types  e.g. [:bpi, :bpi, :argmax],
                maxiters = 100,
                ϵ = 1e-4,                      # convergence criterion
                r = 0., rstep = 0.,            # reinforcement parameters for W vars
                ψ = 0.,                        # damping coefficient
                yy = -1.,                      # focusing BP parameter
                h0 = nothing,                  # external field
                ρ = 1.,                        # coefficient for external field from mini-batch posterior
                rbatch = 0.,                   # reinforcement parameter for external field
                freezetop = false,             # freeze top-layer's weights to 1
                teacher = nothing,
                altsolv::Bool = true,
                altconv::Bool = false,
                seed::Int = -1,
                β = Inf,
                density = 1f0,                  # density of fully connected layer
                batchsize = -1,                 # only supported by some algorithms
                epochs = 100,
                ϵinit = 0.,
                plotinfo = 0,
                verbose = 1,
                usecuda = true,
                gpu_id = -1,
                saveres = false,
                exp_folder=""
                )

    usecuda = CUDA.functional() && usecuda
    device =  usecuda ? gpu : cpu
    usecuda && gpu_id >= 0 && device!(gpu_id)
    if seed > 0
        Random.seed!(seed)
        usecuda && CUDA.seed!(seed)
    end

    L = length(K) - 1
    ψ = num_to_vec(ψ, L)
    ρ = num_to_vec(ρ, L)

    xtrain, ytrain = device(xtrain), device(ytrain)
    xtest, ytest = device(xtest), device(ytest)
    dtrain = DataLoader((xtrain, ytrain); batchsize, shuffle=true, partial=false)

    g = FactorGraph(first(dtrain)..., K, ϵinit, layers; β, density, device)
    h0 !== nothing && set_external_fields!(g, h0; ρ, rbatch);
    if teacher !== nothing
        teacher = device.(teacher)
        has_same_size(g, teacher) && set_weight_mask!(g, teacher)
    end
    initrand!(g)
    freezetop && freezetop!(g, 1)
    reinfpar = ReinfParams(r, rstep, yy, ψ)

    if saveres
        res_folder = "results/" * exp_folder * "/"
        ispath(res_folder) || mkpath(res_folder)

        infolayers = string([string(layers[i], K[i+1], "_") for i = 1:length(layers)]...)
        resfile =  "res_cl_nt$(num_tasks)_$(K[1])_" * infolayers
        resfile *= "bs$(batchsize)_rho$(ρ)_r$(r)_damp$(ψ)"
        density < 1 && (resfile *= "_density$(density)")
        resfile *= "_M$(length(ytrain))_ϵinit$(ϵinit)_maxit$(maxiters)_ep$(epochs)"
        resfile *= "_seed$(seed).dat"
        fres = open(res_folder * resfile, "w")
    end

    N = size(xtrain, 1)
    perms = [randperm(N) for _ = 1:num_tasks]

    function errors(g, x, y)
        mean(vec(forward(g, x)) .!= y)
    end
    function report(epoch; t=(@timed 0), converged=0., solved=0., meaniters=0.)
        outf = @sprintf("%i", epoch)
        for k = 1:num_tasks
            outf *= @sprintf(" %g %g", errors(g, device(xtrain[perms[k],:]), device(ytrain)),
                                      errors(g, device(xtest[perms[k],:]), device(ytest)))
        end

        Etrain = mean(vec(forward(g, xtrain)) .!= ytrain) * 100
        Etrain_bayes = bayesian_error(g, xtrain, ytrain) *100
        num_batches = length(dtrain)
        Etest = 100.0
        if ytest !== nothing
            Etest = mean(vec(forward(g, xtest)) .!= ytest) * 100
            Etest_bayes = bayesian_error(g, xtest, ytest) *100
        end

        verbose >= 1 && @printf("Epoch %i (conv=%g, solv=%g <it>=%g): Etrain=%.2f%% Etest=%.2f%%  r=%g rstep=%g ρ=%s  t=%g (layers=%s, bs=%d)\n",
                                epoch, (converged/num_batches), (solved/num_batches), (meaniters/num_batches),
                                Etrain, Etest, reinfpar.r, reinfpar.rstep, ρ, t.time, "$layers", batchsize)

        verbose >= 1 && @printf("\t\t\tEtrainBayes=%.2f%% EtestBayes=%.2f%%\n", Etrain_bayes, Etest_bayes)

        q0s, qWαβs = plot_info(g, 0; verbose)

        if saveres
            # outf = @sprintf("%d %g %g", epoch, Etrain, Etest)
            # for (q0, qWαβ) in zip(q0s, qWαβs)
            #     outf *= @sprintf(" %g %g", mean(q0), mean(qWαβ))
            # end
            # outf *= @sprintf(" %g", t.time)
            # outf *= @sprintf(" %g %g", Etrain_bayes, Etest_bayes)
            println(fres, outf); flush(fres)
        end
        return Etrain
    end

    for n = 1:num_tasks
        xp, yp = device(xtrain[perms[n],:]), device(ytrain)
        xpt, ypt = device(xtest[perms[n], :]), device(ytest)
        dtrain = DataLoader((xp, yp); batchsize, shuffle=true, partial=false)

        report(0)
        for epoch = 1:epochs
            converged = solved = meaniters = 0
            t = @timed for (b, (x, y)) in enumerate(dtrain)
                all(x->x==0, ρ) || set_Hext_from_H!(g, ρ, rbatch)
                set_input_output!(g, x, y)

                it, e, δ = converge!(g; maxiters, ϵ,
                                        reinfpar, altsolv, altconv, plotinfo=0,
                                        teacher, verbose=verbose-1)
                converged += (δ < ϵ)
                solved    += (e == 0)
                meaniters += it

                verbose >= 2 && print("b = $b / $(length(dtrain))\r")

            end
            Etrain = report(epoch; t, converged, solved, meaniters)
        end
    end
    if saveres
        close(fres)
        println("outfile: $resfile")
        conf_file = "results/conf$(resfile[12:end-4]).jld2"
        @show conf_file
        #save(conf_file, Dict("weights" => getW(g)))
    end

    Etrain = sum(vec(forward(g, xtrain)) .!= ytrain)
    # return g, getW(g), teacher, Etrain, it, conf_file
    return g, getW(g), teacher, Etrain
end

# Mnist - Fashion CL experiment
function solveMF(xtrain::AbstractMatrix, ytrain::AbstractVector,
                xtrain2::AbstractMatrix, ytrain2::AbstractVector;
                num_tasks = 6,
                xtest = nothing, ytest = nothing,
                xtest2 = nothing, ytest2 = nothing,
                dataset = :fashion,
                K::Vector{Int},                # List of widths for each layer, e.g. [28*28, 101, 101, 1]
                layers,                        # List of layer types  e.g. [:bpi, :bpi, :argmax],
                maxiters = 100,
                ϵ = 1e-4,                      # convergence criterion
                r = 0., rstep = 0.,            # reinforcement parameters for W vars
                ψ = 0.,                        # damping coefficient
                yy = -1.,                      # focusing BP parameter
                h0 = nothing,                  # external field
                ρ = 1.,                        # coefficient for external field from mini-batch posterior
                rbatch = 0.,                   # reinforcement parameter for external field
                freezetop = false,             # freeze top-layer's weights to 1
                teacher = nothing,
                altsolv::Bool = true,
                altconv::Bool = false,
                seed::Int = -1,
                β = Inf,
                density = 1f0,                  # density of fully connected layer
                batchsize = -1,                 # only supported by some algorithms
                epochs = 100,
                ϵinit = 0.,
                plotinfo = 0,
                verbose = 1,
                usecuda = true,
                gpu_id = -1,
                saveres = false,
                exp_folder=""
                )

    usecuda = CUDA.functional() && usecuda
    device =  usecuda ? gpu : cpu
    usecuda && gpu_id >= 0 && device!(gpu_id)
    if seed > 0
        Random.seed!(seed)
        usecuda && CUDA.seed!(seed)
    end

    L = length(K) - 1
    ψ = num_to_vec(ψ, L)
    ρ = num_to_vec(ρ, L)

    # First task [MNIST]
    xtrain, ytrain = device(xtrain), device(ytrain)
    xtest, ytest = device(xtest), device(ytest)
    dtrain = DataLoader((xtrain, ytrain); batchsize, shuffle=true, partial=false)

    # Second task [FASHION]
    xtrain2, ytrain2 = device(xtrain2), device(ytrain2)
    xtest2, ytest2 = device(xtest2), device(ytest2)
    dtrain2 = DataLoader((xtrain2, ytrain2); batchsize, shuffle=true, partial=false)

    g = FactorGraph(first(dtrain)..., K, ϵinit, layers; β, density, device)
    h0 !== nothing && set_external_fields!(g, h0; ρ, rbatch);
    if teacher !== nothing
        teacher = device.(teacher)
        has_same_size(g, teacher) && set_weight_mask!(g, teacher)
    end
    initrand!(g)
    freezetop && freezetop!(g, 1)
    reinfpar = ReinfParams(r, rstep, yy, ψ)

    if saveres
        res_folder = "results/" * exp_folder * "/"
        ispath(res_folder) || mkpath(res_folder)

        infolayers = string([string(layers[i], K[i+1], "_") for i = 1:length(layers)]...)
        resfile =  "res_cl_MnistFashion_$(K[1])_" * infolayers
        resfile *= "bs$(batchsize)_rho$(ρ)_r$(r)_damp$(ψ)"
        density < 1 && (resfile *= "_density$(density)")
        resfile *= "_M$(length(ytrain))_ϵinit$(ϵinit)_maxit$(maxiters)_ep$(epochs)"
        resfile *= "_seed$(seed).dat"
        fres = open(res_folder * resfile, "w")
    end

    # N = size(xtrain, 1)
    # perms = [randperm(N) for _ = 1:num_tasks]

    function errors(g, x, y)
        mean(vec(forward(g, x)) .!= y)
    end
    function report(epoch; t=(@timed 0), converged=0., solved=0., meaniters=0.)
        Etrain = mean(vec(forward(g, xtrain)) .!= ytrain) * 100
        Etrain_bayes = bayesian_error(g, xtrain, ytrain) *100
        Etrain2 = mean(vec(forward(g, xtrain2)) .!= ytrain2) * 100
        Etrain2_bayes = bayesian_error(g, xtrain2, ytrain2) *100

        num_batches = length(dtrain)
        Etest = 100.0
        if ytest !== nothing
            Etest = mean(vec(forward(g, xtest)) .!= ytest) * 100
            Etest_bayes = bayesian_error(g, xtest, ytest) *100
            Etest2 = mean(vec(forward(g, xtest2)) .!= ytest2) * 100
            Etest2_bayes = bayesian_error(g, xtest2, ytest2) *100
        end

        verbose >= 1 && @printf("Epoch %i (conv=%g, solv=%g <it>=%g): E1=%.2f%% Et1=%.2f%% E2=%.2f%% Et2=%.2f%% r=%g rstep=%g ρ=%s  t=%g (layers=%s, bs=%d)\n",
                                epoch, (converged/num_batches), (solved/num_batches), (meaniters/num_batches),
                                Etrain, Etest, Etrain2, Etest2, reinfpar.r, reinfpar.rstep, ρ, t.time, "$layers", batchsize)

        verbose >= 1 && @printf("\t\t\tE1B=%.2f%% E1tBayes=%.2f%%\nE2B=%.2f%% E2tBayes=%.2f%%\n",
                                Etrain_bayes, Etest_bayes, Etrain2_bayes, Etest2_bayes)

        q0s, qWαβs = plot_info(g, 0; verbose)

        if saveres
            outf = @sprintf("%d %g %g %g %g", epoch, Etrain, Etest, Etrain2, Etest2)
            # for (q0, qWαβ) in zip(q0s, qWαβs)
            #     outf *= @sprintf(" %g %g", mean(q0), mean(qWαβ))
            # end
            # outf *= @sprintf(" %g", t.time)
            outf *= @sprintf(" %g %g %g %g", Etrain_bayes, Etest_bayes, Etrain2_bayes, Etest2_bayes)
            println(fres, outf); flush(fres)
        end
        return Etrain
    end

    # First Task [Mnist]
    report(0)
    for epoch = 1:epochs
        converged = solved = meaniters = 0
        t = @timed for (b, (x, y)) in enumerate(dtrain)
            all(x->x==0, ρ) || set_Hext_from_H!(g, ρ, rbatch)
            set_input_output!(g, x, y)

            it, e, δ = converge!(g; maxiters, ϵ,
                                    reinfpar, altsolv, altconv, plotinfo=0,
                                    teacher, verbose=verbose-1)
            converged += (δ < ϵ)
            solved    += (e == 0)
            meaniters += it

            verbose >= 2 && print("b = $b / $(length(dtrain))\r")
        end
        Etrain = report(epoch; t, converged, solved, meaniters)
    end

    # Second Task [Fashion]
    for epoch = (epochs+1):(2*epochs)
        converged = solved = meaniters = 0
        t = @timed for (b, (x, y)) in enumerate(dtrain2)
            all(x->x==0, ρ) || set_Hext_from_H!(g, ρ, rbatch)
            set_input_output!(g, x, y)

            it, e, δ = converge!(g; maxiters, ϵ,
                                    reinfpar, altsolv, altconv, plotinfo=0,
                                    teacher, verbose=verbose-1)
            converged += (δ < ϵ)
            solved    += (e == 0)
            meaniters += it

            verbose >= 2 && print("b = $b / $(length(dtrain2))\r")
        end
        Etrain = report(epoch; t, converged, solved, meaniters)
    end

    if saveres
        close(fres)
        println("outfile: $resfile")
        conf_file = "results/conf$(resfile[12:end-4]).jld2"
        @show conf_file
        save(conf_file, Dict("weights" => getW(g)))
    end

    Etrain = sum(vec(forward(g, xtrain)) .!= ytrain)
    # return g, getW(g), teacher, Etrain, it, conf_file
    return g, getW(g), teacher, Etrain
end


end #module
