using MLDatasets: MNIST
using DeepMP
using Test

function get_mnist(M=60000)
    datadir = joinpath(homedir(), "Datasets", "MNIST")
    xtrain, ytrain = MNIST.traindata(Float64, dir=datadir)
    xtest, ytest = MNIST.testdata(Float64, dir=datadir)
    xtrain = reshape(xtrain, :, 60000)
    xtest = reshape(xtest, :, 10000)
    ytrain = map(x-> isodd(x) ? 1 : -1, ytrain)
    ytest = map(x-> isodd(x) ? 1 : -1, ytest)
    return xtrain[:,1:M], ytrain[1:M], xtest, ytest
end

function run_experiment(i)
    if i == 1
        @testset "BP on PERCEPTRON" begin
        M = 100
        xtrain, ytrain, xtest, ytest = get_mnist(M)
        K = [28*28, 1]

        DeepMP.solve(xtrain, ytrain, 
            xtest=xtest, ytest=ytest,
            K = K,
            maxiters=1000,
            r = 0.9, rstep=0.01,
            layers=[:bp])
        end
    elseif i == 2
        @testset "SBP on PERCEPTRON" begin
        M = 300 # 
        xtrain, ytrain, xtest, ytest = get_mnist(M)
        K = [28*28, 1]
        
        batchsize = 1
        DeepMP.solve(xtrain, ytrain, 
            xtest=xtest, ytest=ytest,
            K = K,
            maxiters=100,
            r = 0., rstep=0.,
            batchsize=1, epochs = 50,
            altsolv =false, altconv=true, 
            ρ = 1, 
            layers=[:bp])
        
        batchsize = 10
        DeepMP.solve(xtrain, ytrain, 
            xtest=xtest, ytest=ytest,
            K = K,
            maxiters=100,
            r = 0., rstep=0.,
            batchsize=10, epochs = 50,
            altsolv =false, altconv=true, 
            ρ = 1, 
            layers=[:bp])
        end
    elseif i == 3
        @testset "STAP on PERCEPTRON" begin
        # nessuno va ad errore 0
        M = 200
        xtrain, ytrain, xtest, ytest = get_mnist(M)
        K = [28*28, 1]
        
        batchsize = 1
        DeepMP.solve(xtrain, ytrain, 
            xtest=xtest, ytest=ytest,
            K = K,
            maxiters=100,
            r = 0., rstep=0.,
            batchsize=batchsize, epochs = 50,
            altsolv =false, altconv=true, 
            ρ = 1, 
            layers=[:tap])
        
        batchsize = 10
        DeepMP.solve(xtrain, ytrain, 
            xtest=xtest, ytest=ytest,
            K = K,
            maxiters=10,
            r = 0., rstep=0.,
            batchsize=batchsize, epochs = 50,
            altsolv =false, altconv=true, 
            ρ = 1, 
            layers=[:tap])
        end
    elseif i == 4
        @testset "SBP accurate on PERCEPTRON" begin
        M = 300 # 
        xtrain, ytrain, xtest, ytest = get_mnist(M)
        K = [28*28, 1]
        
        batchsize = 1
        DeepMP.solve(xtrain, ytrain, 
            xtest=xtest, ytest=ytest,
            K = K,
            maxiters=100,
            r = 0., rstep=0.,
            batchsize=1, epochs = 50,
            altsolv =false, altconv=true, 
            ρ = 1, 
            layers=[:bpacc])
        end
    elseif i == 5
        @testset "SBP on COMMETTEE" begin
        # NOT WORKING!!!
        M = 1000
        xtrain, ytrain, xtest, ytest = get_mnist(M)
        K = [28*28, 7, 1]
        
        batchsize = 1
        DeepMP.solve(xtrain, ytrain, 
            xtest=xtest, ytest=ytest,
            K = K,
            maxiters=100,
            r = 0., rstep=0.,
            batchsize=batchsize, epochs = 50,
            altsolv =false, altconv=true,
            ρ = 10., 
            layers=[:bpacc, :bpacc],
            density = [0.5, 1.] 
            )
        end
    end
end
