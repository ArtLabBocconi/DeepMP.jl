using MLDatasets: MNIST, FashionMNIST
using DeepMP
using Test
using Random, Statistics

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

function run_experiment(i)
    if i == 1
        @testset "BP on PERCEPTRON" begin
        M = 100
        xtrain, ytrain, xtest, ytest = get_mnist(M)
        K = [28*28, 1]

        g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain; 
                            xtest, ytest,
                            seed=1,
                            K = K,
                            maxiters=1000,
                            r = 0.8, rstep=0.01,
                            layers=[:bp])
        @test E == 0 
        end#testset
    elseif i == 2
        @testset "SBP on PERCEPTRON" begin
        M = 300 # 
        xtrain, ytrain, xtest, ytest = get_mnist(M)
        K = [28*28, 1]
        
        batchsize = 1
        layers= [:bp]
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

          
        batchsize = 10
        layers= [:bp]
        g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
                            xtest=xtest, ytest=ytest,
                            K = K,
                            maxiters=100,
                            r = 0., rstep=0.,
                            batchsize=10, epochs = 50,
                            altsolv =false, altconv=true, 
                            ρ = 1.001, layers)
        @test_broken E == 0
        end#testset
    elseif i == 3
        @testset "STAP on PERCEPTRON" begin
        # nessuno va ad errore 0
        M = 200
        xtrain, ytrain, xtest, ytest = get_mnist(M)
        K = [28*28, 1]
        
        batchsize = 1
        layers= [:tap]
        g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain; 
                    xtest, ytest,
                    K = K,
                    maxiters=100,
                    r = 0., rstep=0.,
                    batchsize, epochs = 50,
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
        @testset "SBP accurate on PERCEPTRON" begin
        M = 300 # 
        xtrain, ytrain, xtest, ytest = get_mnist(M)
        K = [28*28, 1]
        
        batchsize = 1
        g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain, 
            xtest=xtest, ytest=ytest,
            K = K,
            maxiters=100,
            r = 0., rstep=0.,
            batchsize=1, epochs = 50,
            altsolv =false, altconv=true, 
            ρ = 1, 
            layers=[:bpacc])

        @test E == 0
        end#testset
    elseif i == 5
        @testset "SBP on COMMETTEE" begin
        # NOT WORKING!!!
        M = 1000
        xtrain, ytrain, xtest, ytest = get_mnist(M)
        K = [28*28, 7, 1]
        
        batchsize = 1
        g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
            xtest, ytest,
            K = K,
            maxiters=100,
            r = 0., rstep=0.,
            batchsize=batchsize, epochs = 50,
            altsolv =false, altconv=true,
            ρ = 1., 
            layers=[:bpacc, :bpacc],
            density = [0.5, 1.] 
            )
        end
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
        layers=[:bpacc, :bpacc, :bpacc],

        g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain;
            xtest, ytest,
            K = K,
            seed=1,
            maxiters=10,
            r = 0., rstep=0.,
            batchsize=batchsize, epochs = 50,
            altsolv =false, altconv=true,
            ρ = 1.0, layers,
            density = [0.5, 0.5, 1])
        end
    end
end
