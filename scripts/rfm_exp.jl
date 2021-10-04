using Pkg
Pkg.activate("../")
Pkg.instantiate()

using DeepMP
using Random, Statistics, LinearAlgebra, DelimitedFiles, Printf
using CUDA

function RFMPatterns(D, N, P)
    # ξtrain, ytrain = RandomPatterns(θ, P)
    Ptest = 2 * P
    θ = rand([-1,1], 1, D)
    ξtrain = rand([-1,1], D, P)
    ytrain = sign.(θ * ξtrain .+ 1e-10) |> vec
    ξtest = rand([-1,1], D, Ptest)
    ytest = sign.(θ * ξtest .+ 1e-10) |> vec

    F = rand([-1.0, 1.0], D, N)
    xtrain = sign.(F' * ξtrain .+ 1e-10)
    xtest  = sign.(F' * ξtest  .+ 1e-10)

    return xtrain, ytrain, xtest, ytest
end

function runexp(D, P; inv_arange=LinRange(0.5, 5, 15),
                      outpath = "results/rfm/",
                      outfile = "",
                      K = [101,1],
                      seed_data=13,
                      kws...)

    println("(D, P) = ($D, $P)")
    println("N | 1/α | Etrain | Etest")

    ispath(outpath) || mkpath(outpath)
    if isempty(outfile)
        ai = inv_arange[1]
        af = inv_arange[end]
        da = length(inv_arange)
        outfile = "rfm_D$(D)_P$(P)_" *
                  "ai$(ai)_af$(af)_da$(da)_" *
                  "sx$(seed_data).dat"
    end
    f = open(outpath * outfile, "w")

    seed_data > 0 && Random.seed!(seed_data)

    for inv_a in inv_arange
        N = round(Int, inv_a * P)
        N += Int(iseven(N))
        xtrain, ytrain, xtest, ytest = RFMPatterns(D, N, P)
        g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
                                            xtest, ytest,
                                            K = [N, 1],
                                            kws...)

        Etrain = mean(vec(DeepMP.forward(g, xtrain)) .!= ytrain)
        Etest = mean(vec(DeepMP.forward(g, xtest)) .!= ytest)

        out = @sprintf("%i %f %f %f", N, inv_a, Etrain, Etest)
        println(out)
        println(f, out)
    end
    close(f)
end
