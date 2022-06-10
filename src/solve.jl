
function converge!(g::FactorGraph;  maxiters=10000, ϵ=1f-5,
                                    batchsize=1, saveres=false, fres=nothing,
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
        xtrain, ytrain = g.layers[1].x, g.layers[end].y
        E = mean(vec(forward(g, xtrain)) .!= ytrain) * 100

        verbose >= 1 && @printf("it=%d \t (r=%s, ψ=%s) Etrain=%.2f%% \t Δ=%f \n",
                                 it, reinfpar.r, reinfpar.ψ, E, Δ)

        if verbose >= 2 || (batchsize == -1 && saveres && !isnothing(fres))
            Etest = 100.0
            Etest_bayes = 100.0
            if ytest !== nothing
                Etest = mean(vec(forward(g, xtest)) .!= ytest) * 100
                Etest_bayes = bayesian_error(g, xtest, ytest) *100
            end
            @printf("\t\t\t\t  Etest=%.2f%%   rstep=%g  t=%g\n", Etest, reinfpar.rstep, t.time)
            
            Etrain_bayes = bayesian_error(g, xtrain, ytrain) *100
            @printf("\t  EtrainBayes=%.2f%% EtestBayes=%.2f%%\n", Etrain_bayes, Etest_bayes)
        end

        if plotinfo > 0 || (batchsize == -1 && saveres && !isnothing(fres))
            q0s, qWαβs = plot_info(g, 0; verbose)
        end

        if batchsize == -1 && saveres && !isnothing(fres)
            outf = @sprintf("%d %g %g", it, E, Etest)
            for (q0, qWαβ) in zip(q0s, qWαβs)
                outf *= @sprintf(" %g %g", mean(q0), mean(qWαβ))
            end
            outf *= @sprintf(" %g", Δ)
            outf *= @sprintf(" %g", t.time)
            outf *= @sprintf(" %g %g", Etrain_bayes, Etest_bayes)
            println(fres, outf); flush(fres)
        end

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

"""
    generate_problem(args...; kws...)

Generate a synthetic classification problem, either with random label or
with labels generated by a teacher. 
"""
function generate_problem(; 
                 N = 101,        # number of input features
                 Mtrain = 10000, # num train examples
                 Mtest = 10000, # num test samples
                 TS = false,    # teacher student setting 
                 Kteacher = [N, 3],   # teacher layers' widths
                 density_teacher = 1,  # density on non-zero teacher's weights
                 hidden_manifold = false)  # use the hidden manifold model

    
    D = Kteacher[1]
    @assert hidden_manifold || N == D 
    xtrain = rand(F[-1, 1], D, Mtrain)
    
    if TS
        teacher = rand_teacher(Kteacher; density=density_teacher)
        ytrain = Int.(forward(teacher, xtrain) |> vec)
        xtest = rand(F[-1, 1], D, Mtest)
        ytest = Int.(forward(teacher, xtest) |> vec)
    else
        teacher = nothing
        ytrain = rand([-1,1], Mtrain)
        xtest, ytest = nothing, nothing
    end

    if hidden_manifold
        features = rand(F[-1, 1], N, D)    
        xtrain = sign.(features * xtrain ./ sqrt(D))
        if xtest !== nothing
            xtest = sign.(features * xtest ./ sqrt(D))
        end
    end

    @assert size(xtrain) == (N, Mtrain)
    @assert size(ytrain) == (Mtrain,)
    @assert all(x -> x == -1 || x == 1, ytrain)

    return (; xtrain, ytrain,  teacher, xtest, ytest) 
end

"""
    solve(...)

Solve by message passing a given problem instance.
"""
function solve(xtrain::AbstractMatrix, ytrain::AbstractVector;
                xtest = nothing, ytest = nothing,
                dataset = :fashion,
                K::Vector{Int},                # List of widths for each layer, e.g. [28*28, 101, 101, 1]
                layers,                        # List of layer types  e.g. [(:bpi, :sign), (:bpi, :sign), :argmax],
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
    r = num_to_vec(r, L)

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

        resfile = "results/res_dataset$(dataset)_"
        resfile *= "Ks$(K)_bs$(batchsize)_layers$(layers[1])_rho$(ρ)_r$(r)_damp$(ψ)"
        resfile *= "_density$(density)"
        resfile *= "_M$(length(ytrain))_ϵinit$(ϵinit)_maxiters$(maxiters)"
        seed ≠ -1 && (resfile *= "_seed$(seed)")
        resfile *= ".dat"
        fres = open(resfile, "w")

        #resfile = "results/res_dataset$(dataset)_"
        #resfile *= "Ks$(length(K)-2)x$(K[2])_bs$(batchsize)_layers$(layers[1])_rho$(ρ[1])_r$(r[1])_damp$(ψ[1])"
        #resfile *= "_density$(density[1])"
        #resfile *= "_M$(length(ytrain))_ϵinit$(ϵinit)_maxiters$(maxiters)"
        #seed ≠ -1 && (resfile *= "_seed$(seed)")
        #resfile *= ".dat"
        #fres = open(resfile, "w")

    else
        fres = nothing
    end
    
    function report(epoch; t=(@timed 0), converged=0., solved=0., meaniters=0.)

        Etrain = mean(vec(forward(g, xtrain)) .!= ytrain) * 100
        Etrain_bayes = bayesian_error(g, xtrain, ytrain) * 100
        num_batches = length(dtrain)
        Etest, Etest_bayes = 100.0, 100
        if ytest !== nothing
            Etest = mean(vec(forward(g, xtest)) .!= ytest) * 100
            Etest_bayes = bayesian_error(g, xtest, ytest) * 100
        end
        
        verbose >= 1 && @printf("Epoch %i (conv=%g, solv=%g <it>=%g): Etrain=%.2f%% Etest=%.2f%%  r=%s rstep=%g ρ=%s ψ=%s  t=%g (layers=%s, bs=%d)\n",
                                epoch, (converged/num_batches), (solved/num_batches), (meaniters/num_batches),
                                Etrain, Etest, reinfpar.r, reinfpar.rstep, ρ, ψ, t.time, "$layers", batchsize)
            
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
        it, e, δ = converge!(g; maxiters, 
                            batchsize, saveres, fres, ϵ, reinfpar,
                            altsolv, altconv, plotinfo=1,
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
                reset_downgoing_messages!(g)

                it, e, δ = converge!(g; maxiters, ϵ, 
                                        reinfpar, altsolv, altconv, plotinfo=0,
                                        teacher, verbose=verbose-1)
                converged += (δ < ϵ)
                solved    += (e == 0)
                meaniters += it
                
                verbose >= 2 && print("b = $b / $(length(dtrain))\r")
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
        save(conf_file, Dict("graph" => Array{F}(g.layers[2].m)))
        #save(conf_file, Dict("weights" => getW(g)))
        if !all(x->x==1.0, density)
            for l=2:g.L+1
                file = "results/mask_K$(K[2])_density$(density)_layer$(l-1)_seed$(seed).dat"
                writedlm(file, g.layers[l].weight_mask)
                println(file)
            end
        end
    else
        conf_file = nothing
    end

    Etrain = sum(vec(forward(g, xtrain)) .!= ytrain)
    return g, getW(g), teacher, Etrain, it, conf_file

end