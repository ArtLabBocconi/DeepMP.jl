using Test, DelimitedFiles

type = :bpi

# Odd vs Even or 1 class vs another
function get_mnist(M=60000; classes=[], seed=17, fashion=false)
    datadir = joinpath(homedir(), "Datasets", "MNIST")
    Dataset = fashion ? FashionMNIST : MNIST
    xtrain, ytrain = Dataset.traindata(Float64, dir=datadir)
    xtest, ytest = Dataset.testdata(Float64, dir=datadir)
    xtrain = reshape(xtrain, :, 60000)
    xtest = reshape(xtest, :, 10000)
    if !isempty(classes)
        @assert length(classes) == 2
        seed > 0 && Random.seed!(seed)
        filter = x -> x==classes[1] || x==classes[2]
        idxtrain = findall(filter, ytrain) |> shuffle
        idxtest = findall(filter, ytest) |> shuffle
        xtrain = xtrain[:, idxtrain]
        xtest = xtest[:, idxtest] 
        ytrain = ytrain[idxtrain]
        ytest = ytest[idxtest]
        relabel = x -> x == classes[1] ? 1 : -1
        ytrain = map(relabel, ytrain)
        ytest = map(relabel, ytest)
    else
        ytrain = map(x-> isodd(x) ? 1 : -1, ytrain)
        ytest = map(x-> isodd(x) ? 1 : -1, ytest)
    end
    M = min(M, length(ytrain))
    xtrain, ytrain = xtrain[:,1:M], ytrain[1:M]
    return xtrain, ytrain, xtest, ytest
end


# @testset "Perceptron" begin

#     @time g, W, teacher, E = DeepMP.solve(α=0.4, K=[201,1], 
#                     layers=[type], verbose=1,
#                     altsolv=true, altconv=false,
#                     r=0.2, rstep=0.01, 
#                     seedx=1,maxiters=1000);
#     @test E == 0


#     @time g, W, teacher, E = DeepMP.solve(α=0.5, K=[201,1], 
#                     layers=[type], verbose=1,
#                     altsolv=true, altconv=false,
#                     r=0.2, rstep=0.01, 
#                     seedx=1,maxiters=1000);
    
#     if type == :bpi
#         @test E == 0
#     else
#         @test_broken E == 0
#         @test E <= 5
#     end

#     @time g, W, teacher, E = DeepMP.solve(α=0.5, K=[201,1], 
#                     layers=[type], verbose=1,
#                     batchsize=1, 
#                     ρ=0.8,
#                     r=0.2, rstep=0., 
#                     epochs=100, maxiters=10,
#                     altsolv=false, altconv=true,
#                     seedx=1);
#     @test E == 0
    
#     # FOCUSING # TO FIX
#     # @time g, W, teacher, E = DeepMP.solve(α=0.5, K=[201,1], 
#     #                 layers=[type], verbose=1,
#     #                 batchsize=1, 
#     #                 ρ=0.8,
#     #                 yy=5, r=0.2, rstep=0.01, 
#     #                 epochs=100, maxiters=10,
#     #                 altsolv=false, altconv=true,
#     #                 seedx=1);
#     # @test E == 0

# end#testset

# @testset "Perceptron MNIST" begin
#     M = 300 # 
#     xtrain, ytrain, xtest, ytest = get_mnist(M)
#     K = [28*28, 1]
    
#     batchsize = 1
#     g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain; 
#                         xtest, ytest,
#                         K, verbose=1,
#                         ρ=1, r=0., rstep=0.,
#                         batchsize, 
#                         epochs = 50, maxiters=10, 
#                         altsolv=false, altconv=true, 
#                         layers=[type])
#     if type == :bpi
#         @test_broken E == 0
#         @test E < 5
#     else
#         @test E == 0
#     end

#     batchsize = 10
#     g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
#                         xtest=xtest, ytest=ytest,
#                         K = K,
#                         maxiters=100, verbose=1,
#                         r = 0., rstep=0.,
#                         batchsize, epochs = 50,
#                         altsolv =false, altconv=true, 
#                         ρ = 1., layers=[type])
#     @test_broken E == 0
#     @test_broken E < 10
#     @test E < 50
# end#testset

@testset "MNIST MLP" begin
    y = Int.(readdlm(@__DIR__() * "/../fmnist/seed7/Y.txt")) |> vec
    X = readdlm(@__DIR__() * "/../fmnist/seed7/X.txt")

    batchsize = 1
    ρ, r = 0.8, 0.2
    layers=[type, type, type]

    @time g, W, teacher, E = DeepMP.solve(X,y; K=[784,31,31,1],
                    layers, verbose=1,
                    ρ, r, rstep=0, 
                    epochs=100, maxiters=10,
                    seed=2, density=0.5,
                    batchsize, altsolv=false, altconv=true);

    @test_broken E == 0
    
    batchsize = 10
    ρ, r = 0.8, 0.2
    layers=[type, type, type]

    @time g, W, teacher, E = DeepMP.solve(X,y; K=[784,31,31,1],
                    layers, verbose=1,
                    ρ, r, rstep=0, 
                    epochs=100, maxiters=10,
                    seed=2, density=0.5,
                    batchsize, altsolv=false, altconv=true);

    @test_broken E == 0

    ########### FASHION ##############
    M = 2000
    xtrain, ytrain, xtest, ytest = get_mnist(M, fashion=true)
    K = [28*28, 15, 15, 1]

    batchsize = 10
    ρ, r = 0.9, 0.1 
    layers=[type, type, type]

    g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
                xtest, ytest, K,
                seed=1, verbose=1,
                epochs=50, maxiters=10, 
                ρ, r, rstep=0,
                batchsize, altsolv=false, altconv=true,
                layers, density=[0.5, 0.5, 1])

    @test_broken E == 0

    ########### MNIST ##############
    
    M = 1000
    xtrain, ytrain, xtest, ytest = get_mnist(M)
    K = [28*28, 7, 1]
    
    batchsize = 1
    ρ, r = 0.9, 0.1 

    layers=[type, type]
    g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
        xtest, ytest, verbose=1,
        K, maxiters=100,
        r, ρ, 
        freezetop=true, rstep=0.,
        batchsize, epochs = 50,
        altsolv =false, altconv=true,
        layers, density = [0.5, 1.] 
        )
    
    @test_broken E < 5
end#testset
