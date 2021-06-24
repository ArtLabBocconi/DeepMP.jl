# ONLINE Training: Message Passing vs Gradient Descent

module runexp

using Statistics, Random, LinearAlgebra, DelimitedFiles

# include("../../src/DeepMP.jl")
# include("../../../DeepBinaryNets/src/DeepBinaryNets.jl")
include("../../scripts/real_data_experiments.jl")

function error_sgd(w, x, y)
    return mean(vec(sign.(DeepBinaryNets.forward(w, x))) .!= y)
end
function error_bp(g, x, y)
    return mean(vec(DeepMP.forward(g, x)) .!= y)
end

# MNIST + FASHION MNIST
function deepmp_scenario1(;
                          H::Vector{Int}=[101, 101], epochs=10,
                          layers=[:tap, :tap, :argmax], maxiters=1,
                          ρ=1.0, r=0.0, rstep=0.0, ψ=0.0, ϵinit=1.0,
                          altsolv=true, altconv=true,
                          usecuda=false, gpu_id=1,
                          outfile="tmp.dat", seed=23)

    f = open(outfile, "w")

    xM, yM, xMt, yMt = get_dataset(multiclass=true, dataset=:mnist)
    xF, yF, xFt, yFt = get_dataset(multiclass=true, dataset=:mnist)

    # needed to initaliaze g
    g, wb, wt, E, it = DeepMP.solve(xM, yM;
                 K=[size(x, 1), H..., 10], layers=layers,
                 xtest=xMt, ytest=yMt, ϵinit=ϵinit,
                 ρ=ρ, r=r, rstep=rstep, yy=0.0,
                 seed=seed, epochs=1, maxiters=maxiters,
                 ψ=ψ, density=density, batchsize=1,
                 ϵ=1e-4, altsolv=altsolv, altconv=altconv,
                 freezetop=false,
                 usecusa=usecuda, gpu_id=gpu_id,
                 infotime=1, verbose=1);


    train_err_mnist = error_bp(g, xM, yM)
    test_err_mnist = error_bp(g, xMt, yMt)
    train_err_fashion = error_bp(g, xF, yF)
    test_err_fashion = error_bp(g, xFt, yFt)
    println(f, "$(1) $train_err_mnist $test_err_mnist $train_err_fashion $test_err_fashion")

    # MNIST training
    for ep = 1:(div(epochs-1, 2))
        g, wb, wt, E, it = DeepMP.solve(xM, yM; g0=g,
                     K=[size(x, 1), H..., 10], layers=layers,
                     xtest=nothing, ytest=nothing, ϵinit=ϵinit,
                     ρ=ρ, r=r, rstep=rstep, yy=0.0,
                     seed=seed, epochs=2, maxiters=maxiters,
                     ψ=ψ, density=density, batchsize=1,
                     ϵ=1e-4, altsolv=altsolv, altconv=altconv,
                     freezetop=false,
                     usecusa=usecuda, gpu_id=gpu_id,
                     infotime=1, verbose=1);

        println("ep=$ep")
        train_err_mnist = error_bp(g, xM, yM)
        test_err_mnist = error_bp(g, xMt, yMt)
        train_err_fashion = error_bp(g, xF, yF)
        test_err_fashion = error_bp(g, xFt, yFt)
        println(f, "$(1+ep*2) $train_err_mnist $test_err_mnist $train_err_fashion $test_err_fashion")
    end

    # FASHION training
    for ep = 1:(div(epochs-1, 2))
        g, wb, wt, E, it = DeepMP.solve(xF, yF; g0=g,
                     K=[size(x, 1), H..., 10], layers=layers,
                     xtest=nothing, ytest=nothing, ϵinit=ϵinit,
                     ρ=ρ, r=r, rstep=rstep, yy=0.0,
                     seed=seed, epochs=2, maxiters=maxiters,
                     ψ=ψ, density=density, batchsize=1,
                     ϵ=1e-4, altsolv=altsolv, altconv=altconv,
                     freezetop=false,
                     usecusa=usecuda, gpu_id=gpu_id,
                     infotime=1, verbose=1);

        println("ep=$ep")
        train_err_mnist = error_bp(g, xM, yM)
        test_err_mnist = error_bp(g, xMt, yMt)
        train_err_fashion = error_bp(g, xF, yF)
        test_err_fashion = error_bp(g, xFt, yFt)
        println(f, "$(epochs+ep*2) $train_err_mnist $test_err_mnist $train_err_fashion $test_err_fashion")
    end

    close(f)
end # scenario1

end # module