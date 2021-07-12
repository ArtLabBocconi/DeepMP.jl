using Pkg
Pkg.activate("../")
#Pkg.activate("./")
Pkg.instantiate()

using MLDatasets: MNIST, FashionMNIST, CIFAR10, CIFAR100
using DeepMP
using Test
using Random, Statistics
#using ProfileView
using CUDA

# Odd vs Even or 1 class vs another
function get_dataset(M=-1; multiclass=false, classes=[], seed=17, dataset=:mnist, normalize=true)
    seed > 0 && Random.seed!(seed)
    namedir, Dataset, reduce_dims  = dataset == :fashion ? ("FashionMNIST", FashionMNIST, (1,2,3)) :
                                     dataset == :mnist   ? ("MNIST", MNIST, (1,2,3)) :
                                     dataset == :cifar10 ? ("CIFAR10", CIFAR10, (1,2,4)) :
                                     dataset == :cifar100 ? ("CIFAR100", CIFAR100, (1,2,4)) : 
                                     error("uknown dataset")
    
    datadir = joinpath(homedir(), "Datasets", namedir)
    xtrain, ytrain = Dataset.traindata(DeepMP.F, dir=datadir)
    xtest, ytest = Dataset.testdata(DeepMP.F, dir=datadir)
    @assert all(isinteger.(ytest))
    if normalize
        mn = mean(xtrain, dims=reduce_dims)
        st = std(xtrain, dims=reduce_dims)
        xtrain = (xtrain .- mn) ./ (st .+ 1e-5)
        xtest = (xtest .- mn) ./ (st .+ 1e-5)
    end
    xtrain = reshape(xtrain, :, size(xtrain)[end])
    xtest = reshape(xtest, :, size(xtest)[end])
    if !isempty(classes)
        # ONE CLASS VS ANOTHER
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
    elseif !multiclass
        # ODD VS EVEN        
        ytrain = map(x-> isodd(x) ? 1 : -1, ytrain)
        ytest = map(x-> isodd(x) ? 1 : -1, ytest)
        idxtrain = 1:length(ytrain) |> shuffle
        xtrain = xtrain[:, idxtrain]
        ytrain = ytrain[idxtrain]
    else
        # MULTICLASS CLASSIFICATION
        # set labels always in 1:K
        classes = ytest |> unique |> sort
        class_map = Dict(v => k for (k, v) in enumerate(classes))
        ytrain = map(y -> class_map[y], ytrain)
        ytest = map(y -> class_map[y], ytest)   
    end

    if M < 0
        M = size(xtrain)[end]
    end
    M = min(M, length(ytrain))
    xtrain, ytrain = xtrain[:,1:M], ytrain[1:M]
    return xtrain, ytrain, xtest, ytest
end

function run_experiment(; M=-1, dataset=:fashion, multiclass=false, kws...)
    xtrain, ytrain, xtest, ytest = get_dataset(M; dataset, multiclass)
    g, w, teacher, E, it = DeepMP.solve(xtrain, ytrain; xtest, ytest, dataset, kws...)

    GC.gc()
    CUDA.reclaim()

end

## Multiclass Experiment
# run_experiment(epochs=100, usecuda=true, maxiters=1, batchsize=128, 
#             layers=[:bpi,:bpi,:argmax], ρ=1+1e-4, ψ=0.9, ϵinit=1.,
#             dataset=:mnist, multiclass=true, K=[28*28,101,101,10]);

