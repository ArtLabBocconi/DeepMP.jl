# ONLINE Training: Message Passing vs Gradient Descent

module runexp

using Statistics, Random, LinearAlgebra, DelimitedFiles, Printf

include("real_data_experiments.jl")

using CUDA, KernelAbstractions, CUDAKernels
using Functors

cpu(x) = fmap(x -> adapt(Array, x), x)

gpu(x) = fmap(CUDA.cu, x)
# CUDA.cu(x::Integer) = x
CUDA.cu(x::Float64) = Float32(x)
CUDA.cu(x::Array{Int64}) = convert(CuArray{Int32}, x)

function error_sgd(w, x, y)
    return mean(vec(sign.(DeepBinaryNets.forward(w, x))) .!= y)
end
function error_bp(g, x, y)
    return mean(vec(DeepMP.forward(g, x)) .!= y)
end

# MNIST + FASHION MNIST [SCENARIO 1]
function deepmp_scenario1(; M=-1, bsize=100,
                          H::Vector{Int}=[101, 101], epochs=10,
                          layers=[:tap, :tap, :argmax], maxiters=1,
                          ρ=1.0, r=0.0, rstep=0.0, ψ=0.0, ϵinit=1.0,
                          altsolv=true, altconv=true,
                          usecuda=false, gpu_id=1,
                          outfile="tmp.dat", seed=23)

    usecuda = CUDA.functional() && usecuda
    device = usecuda ? gpu : cpu
    usecuda && gpu_id >= 0 && device!(gpu_id)

    f = open(outfile, "w")

    xM, yM, xMt, yMt = get_dataset(M; multiclass=true, dataset=:mnist)
    xF, yF, xFt, yFt = get_dataset(M; multiclass=true, dataset=:fashion)

    xM, yM, xMt, yMt = device(xM), device(yM), device(xMt), device(yMt)
    xF, yF, xFt, yFt = device(xF), device(yF), device(xFt), device(yFt)

    # needed to initaliaze g
    g, wb, wt, E, it = DeepMP.solve(xM, yM;
                 K=[size(xM, 1), H..., 10], layers=layers,
                 xtest=xMt, ytest=yMt, ϵinit=ϵinit,
                 ρ=ρ, r=r, rstep=rstep, yy=0.0,
                 seed=seed, epochs=1, maxiters=maxiters,
                 ψ=ψ, density=1, batchsize=bsize,
                 ϵ=1e-4, altsolv=altsolv, altconv=altconv,
                 freezetop=false,
                 usecuda=usecuda, gpu_id=gpu_id,
                 verbose=1);


    train_err_mnist = error_bp(g, xM, yM)
    test_err_mnist = error_bp(g, xMt, yMt)
    train_err_fashion = error_bp(g, xF, yF)
    test_err_fashion = error_bp(g, xFt, yFt)
    println(f, "$(1) $train_err_mnist $test_err_mnist $train_err_fashion $test_err_fashion")

    # MNIST training
    for ep = 1:(div(epochs-1, 2))
        g, wb, wt, E, it = DeepMP.solve(xM, yM; g0=g,
                     K=[size(xM, 1), H..., 10], layers=layers,
                     xtest=nothing, ytest=nothing, ϵinit=ϵinit,
                     ρ=ρ, r=r, rstep=rstep, yy=0.0,
                     seed=seed, epochs=2, maxiters=maxiters,
                     ψ=ψ, density=1, batchsize=bsize,
                     ϵ=1e-4, altsolv=altsolv, altconv=altconv,
                     freezetop=false,
                     usecuda=usecuda, gpu_id=gpu_id,
                     verbose=1);

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
                     K=[size(xM, 1), H..., 10], layers=layers,
                     xtest=nothing, ytest=nothing, ϵinit=ϵinit,
                     ρ=ρ, r=r, rstep=rstep, yy=0.0,
                     seed=seed, epochs=2, maxiters=maxiters,
                     ψ=ψ, density=1, batchsize=bsize,
                     ϵ=1e-4, altsolv=altsolv, altconv=altconv,
                     freezetop=false,
                     usecuda=usecuda, gpu_id=gpu_id,
                     verbose=1);

        println("ep=$ep")
        train_err_mnist = error_bp(g, xM, yM)
        test_err_mnist = error_bp(g, xMt, yMt)
        train_err_fashion = error_bp(g, xF, yF)
        test_err_fashion = error_bp(g, xFt, yFt)
        println(f, "$(epochs+ep*2) $train_err_mnist $test_err_mnist $train_err_fashion $test_err_fashion")
    end

    close(f)
end # scenario1

# Permuted MNIST [SCENARIO 2]
function deepmp_scenario2(; M=-1,  bsize=100,
                          num_tasks=5,
                          H::Vector{Int}=[101, 101], epochs=10,
                          layers=[:tap, :tap, :argmax], maxiters=1,
                          ρ=1.0, r=0.0, rstep=0.0, ψ=0.0, ϵinit=1.0,
                          altsolv=true, altconv=true,
                          usecuda=false, gpu_id=1,
                          outfile="tmp.dat", seed=23)

    usecuda = CUDA.functional() && usecuda
    device = usecuda ? gpu : cpu
    usecuda && gpu_id >= 0 && device!(gpu_id)

    f = open(outfile, "w")

    x, y, xt, yt = get_dataset(M; multiclass=true, dataset=:mnist)

    x, y, xt, yt = device(x), device(y), device(xt), device(yt)

    N = size(x, 1)
    perms = [randperm(N) for _ = 1:num_tasks]
    train_errs = [1.0 for _ = 1:num_tasks]
    test_errs  = [1.0 for _ = 1:num_tasks]

    # 1st task, 1 epoch, needed to initaliaze g
    g, wb, wt, E, it = DeepMP.solve(x[perms[1],:], y;
                    K=[size(x, 1), H..., 10], layers=layers,
                    xtest=xt[perms[1],:], ytest=yt, ϵinit=ϵinit,
                    ρ=ρ, r=r, rstep=rstep, yy=0.0,
                    seed=seed, epochs=1, maxiters=maxiters,
                    ψ=ψ, density=1, batchsize=bsize,
                    ϵ=1e-4, altsolv=altsolv, altconv=altconv,
                    freezetop=false,
                    usecuda=usecuda, gpu_id=gpu_id,
                    verbose=1);


    for n = 1:num_tasks
        for it = 1:div(epochs, 2)
            # solve
            g, wb, wt, E, it = DeepMP.solve(x[perms[n],:], y;
                            g0 = g,
                            K=[size(x, 1), H..., 10], layers=layers,
                            xtest=xt[perms[n],:], ytest=yt, ϵinit=ϵinit,
                            ρ=ρ, r=r, rstep=rstep, yy=0.0,
                            seed=seed, epochs=2, maxiters=maxiters,
                            ψ=ψ, density=1, batchsize=bsize,
                            ϵ=1e-4, altsolv=altsolv, altconv=altconv,
                            freezetop=false,
                            usecuda=usecuda, gpu_id=gpu_id,
                            verbose=1);
            out = @sprintf("%i", it*2)
            for k = 1:num_tasks
                out *= @sprintf(" %g %g", error_bp(g, x[perms[k],:], y), error_bp(g, xt[perms[k],:], yt))
            end
            println(f, out)
        end
    end

    close(f)
end # scenario 2

end # module
