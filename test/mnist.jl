using MLDatasets: MNIST, FashionMNIST
using DeepMP
using Test
using Random, Statistics

@testset "BP on PERCEPTRON" begin
    M = 100
    xtrain, ytrain, xtest, ytest = get_mnist(M)
    K = [28*28, 1]

    layers=[:bp]
    g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain; 
                        xtest, ytest,
                        seed=1,
                        K = K, ψ = 0.,
                        maxiters=100,
                        r = 0.8, rstep=0.01,
                        layers)
    @test_broken E == 0 
    @test E <= 2 
    
    M = 300
    xtrain, ytrain, xtest, ytest = get_mnist(M)
    K = [28*28, 1]

    layers=[:bpacc]
    g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain; 
                        xtest, ytest,
                        seed=1,
                        K = K, ψ = 0.9,
                        maxiters=300,
                        r = 0.2, rstep=0.01,
                        layers)
    @test E == 0 
end#testset

@testset "SBP on PERCEPTRON" begin
    M = 300 # 
    xtrain, ytrain, xtest, ytest = get_mnist(M)
    K = [28*28, 1]
    
    batchsize = 1
    layers= [:bp]
    g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain; 
                        xtest, ytest, K,
                        maxiters=100,
                        ρ=1, r=0., rstep=0.,
                        batchsize, epochs=50,
                        altsolv =false, altconv=true, 
                        layers)
    @test_broken E == 0
    @test E < 25

    batchsize = 10
    layers= [:bp]
    g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
                        xtest=xtest, ytest=ytest,
                        K = K,
                        maxiters=100,
                        ρ=1., r=0., rstep=0.,
                        batchsize, epochs=50,
                        altsolv=false, altconv=true, 
                        layers)
    @test_broken E == 0
    @test_broken E < 10
    @test E < 50

    batchsize = 1
    g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
        xtest, ytest,
        K, maxiters=100,
        r = 0., rstep=0.,
        batchsize, epochs = 50,
        altsolv =false, altconv=true, 
        ρ = 1, 
        layers=[:bpacc])

    @test E == 0
    
    batchsize = 10
    layers= [:bpacc]
    g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
                        xtest=xtest, ytest=ytest,
                        K = K,
                        maxiters=100,
                        r = 0., rstep=0.,
                        batchsize, epochs = 50,
                        altsolv =false, altconv=true, 
                        ρ = 1, layers)
    @test_broken E == 0
    @test E <= 1
    
end

@testset "STAP on PERCEPTRON" begin
    M = 300
    xtrain, ytrain, xtest, ytest = get_mnist(M)
    K = [28*28, 1]

    batchsize = 1
    layers= [:tap]
    g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain; 
                xtest, ytest,
                K, seed=1,
                maxiters=100,
                r = 0., rstep=0.,
                batchsize, epochs = 100,
                altsolv =false, altconv=true, 
                ρ = 1, layers)

    @test E == 0

    batchsize = 10
    layers= [:tap]
    g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
                xtest, ytest,
                K = K,
                maxiters=10,
                r = 0., rstep=0.,
                batchsize, epochs = 50,
                altsolv =false, altconv=true, 
                ρ = 1., layers)

    @test_broken E == 0
end#testset


@testset "SBP on COMMETTEE" begin
        M = 1000
        xtrain, ytrain, xtest, ytest = get_mnist(M)
        K = [28*28, 7, 1]
        
        batchsize = 1
        layers=[:bpacc, :bpacc]
        g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
            xtest, ytest,
            K, maxiters=100,
            r=0, rstep=0.,
            batchsize, epochs = 50,
            altsolv =false, altconv=true,
            ρ=1, freezetop=true,
            layers, density = [0.5, 1.] 
            )
        
        @test E < 5 
        
        M = 1000
        xtrain, ytrain, xtest, ytest = get_mnist(M)
        K = [28*28, 7, 1]
        
        batchsize = 1
        layers=[:tap, :bp]
        g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
            xtest, ytest,
            K, maxiters=100,
            r=0.1, rstep=0.,
            batchsize, epochs = 50,
            altsolv=false, altconv=true,
            ρ=0.9, freezetop=true,
            layers, density = [0.5, 1.] 
            )
        
        @test E < 5
        
        M = 1000
        xtrain, ytrain, xtest, ytest = get_mnist(M)
        K = [28*28, 7, 1]
        
        batchsize = 1
        layers=[:tap, :bp]
        g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
            xtest, ytest,
            K, maxiters=100,
            r=0.9, rstep=0.,
            batchsize, epochs = 50,
            altsolv =false, altconv=true,
            ρ=0.1, freezetop=true,
            layers, density = [0.5, 1.] 
            )
        
        @test E < 5
end#testset
    
@testset "SBP on MLP" begin
    # M = 10000
    # xtrain, ytrain, xtest, ytest = get_mnist(M, fashion=true, 
    #                                     classes=[4,5])
    # K = [28*28, 1]

    # batchsize = 1
    # DeepMP.solve(xtrain, ytrain, 
    #     xtest=xtest, ytest=ytest,
    #     K = K,
    #     maxiters=10,
    #     r = 0., rstep=0.,
    #     batchsize=batchsize, epochs = 50,
    #     altsolv =false, altconv=true,
    #     ρ = 1., 
    #     layers=[:bpacc],
    #     density = 1)
    # end

    M = 2000
    xtrain, ytrain, xtest, ytest = get_mnist(M, fashion=true, 
                                        classes=[4,5])
    K = [28*28, 15, 15, 1]

    batchsize = 10
    layers=[:bpacc, :bpacc, :bpacc]

    g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
        xtest, ytest,
        K = K,
        seed=1,
        maxiters=10,
        r = 0., rstep=0.,
        batchsize, epochs = 50,
        altsolv =false, altconv=true,
        ρ = 1.0, layers,
        density = [0.5, 0.5, 1])

    @test E == 0
end