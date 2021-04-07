using MLDatasets: MNIST, FashionMNIST
using DeepMP
using Test
using Random, Statistics

# Odd vs Even or 1 class vs another
function get_mnist(M=60000; classes=[], seed=17, fashion=false)
    seed > 0 && Random.seed!(seed)
    namedir = fashion ? "FashionMNIST" : "MNIST"
    datadir = joinpath(homedir(), "Datasets", namedir)
    Dataset = fashion ? FashionMNIST : MNIST
    xtrain, ytrain = Dataset.traindata(DeepMP.F, dir=datadir)
    xtest, ytest = Dataset.testdata(DeepMP.F, dir=datadir)
    xtrain = reshape(xtrain, :, 60000)
    xtest = reshape(xtest, :, 10000)
    if !isempty(classes)
        @assert length(classes) == 2
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
        idxtrain = 1:length(ytrain) |> shuffle
        xtrain = xtrain[:, idxtrain]
        ytrain = ytrain[idxtrain]
    end
    M = min(M, length(ytrain))
    xtrain, ytrain = xtrain[:,1:M], ytrain[1:M]
	#@show (ytrain); error()
    return xtrain, ytrain, xtest, ytest
end

function run_experiment(i; M=100, batchsize=1, K = [28*28, 101, 101, 1])
    if i == 7
        #@testset "SBP on MLP" begin

        xtrain, ytrain, xtest, ytest = get_mnist(M, fashion=true, classes=[])
        
        #layers=[:bp, :bp, :bp]
        layers = [:bp for _=1:(length(K)-1)]

        g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
            xtest, ytest,
            K = K,
            seed = 1,
            maxiters = 1,
            r = 0., rstep = 0.,
			ψ = 0.5, yy=-1,
            batchsize, epochs = 100,
            altsolv = false, altconv = true,
            ρ = 1., layers, verbose = 1,
            density = 1)
        #end
        
	elseif i == 8
        
        #@testset "SBP on MLP" begin

        xtrain, ytrain, xtest, ytest = get_mnist(M, fashion=true, classes=[])
        K = [28*28, 101, 1]
        
        batchsize = -1
        layers=[:bp, :bp, :bp]

        g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
            xtest, ytest,
            K = K,
            seed = 1,
            maxiters = 100,
            r = 0.9, rstep = 0.001,
			ψ = 0.3, yy=-1,
            batchsize, epochs = 50,
            altsolv = false, altconv = true,
            ρ = 0., layers, verbose = 2,
            density = 1)
        #en
	elseif i == 1
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
        @test E == 0 
        
        M = 300
        xtrain, ytrain, xtest, ytest = get_mnist(M)
        K = [28*28, 1]

        layers=[:bpacc]
        g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain; 
                            xtest, ytest,
                            seed=1,
                            K = K, ψ = 0.9,
                            maxiters=500,
                            r = 0.2, rstep=0.01,
                            layers)
        @test E == 0 
        end#testset
    elseif i == 2
        @testset "SBP on PERCEPTRON" begin
        M = 300 # 
        xtrain, ytrain, xtest, ytest = get_mnist(M)
        K = [28*28, 1]
        
        batchsize = 1
        layers = [:bp]
        g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain; 
                            xtest, ytest,
                            K = K,
                            maxiters=100,
                            r = 0., rstep=0.,
                            batchsize=1, epochs = 50,
                            altsolv =false, altconv=true, 
                            ρ = 1, 
                            layers)
            
        @test E == 0

        batchsize = 10
        layers= [:bp]
        g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
                            xtest=xtest, ytest=ytest,
                            K = K,
                            maxiters=100,
                            r = 0., rstep=0.,
                            batchsize=10, epochs = 50,
                            altsolv =false, altconv=true, 
                            ρ = 1., layers)
        @test_broken E == 0

        batchsize = 1
        g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
            xtest, ytest,
            K, maxiters=100,
            r = 0., rstep=0.,
            batchsize=1, epochs = 50,
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
                            batchsize=10, epochs = 50,
                            altsolv =false, altconv=true, 
                            ρ = 1, layers)
        @test E == 0
        
        end#testset
    elseif i == 3
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

    elseif i == 4
        @testset "BOH" begin
        
       
        end#testset
    elseif i == 5
        @testset "SBP on COMMITTEE" begin
        M = 1000
        xtrain, ytrain, xtest, ytest = get_mnist(M)
        K = [28*28, 7, 1]
        
        # batchsize = 1
        # layers=[:bpacc, :bpacc]
        # g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
        #     xtest, ytest,
        #     K, maxiters=100,
        #     r=0, rstep=0.,
        #     batchsize, epochs = 50,
        #     altsolv =false, altconv=true,
        #     ρ=1, freezetop=true,
        #     layers, density = [0.5, 1.] 
        #     )
        
        # @test E < 5 
        
        # M = 1000
        # xtrain, ytrain, xtest, ytest = get_mnist(M)
        # K = [28*28, 7, 1]
        
        # batchsize = 1
        # layers=[:tap, :bp]
        # g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
        #     xtest, ytest,
        #     K, maxiters=100,
        #     r=0.1, rstep=0.,
        #     batchsize, epochs = 50,
        #     altsolv =false, altconv=true,
        #     ρ=0.9, freezetop=true,
        #     layers, density = [0.5, 1.] 
        #     )
        
        # @test E < 5
        
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
    
    elseif i == 6
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
        end
    end
end
