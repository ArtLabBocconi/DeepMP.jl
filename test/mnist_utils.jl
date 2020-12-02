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
