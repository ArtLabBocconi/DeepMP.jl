using DeepMP
using Test, DelimitedFiles, Random, Statistics
using MLDatasets: MNIST, FashionMNIST, CIFAR10, CIFAR100
using CUDA

CUDA.allowscalar(false)

include("test_utils.jl")

@testset "MNIST MLP" begin
    include("mnist_mlp.jl")
end

# @testset "Perceptron" begin
#     include("perceptron.jl")
# end

# @testset "Stochastic BP" begin
#     include("stochastic_bp.jl")
# end

# @testset "MLP" begin
#     include("mlp.jl")
# end

# @testset "BPI_MLP" begin
#     include("bpi_mlp.jl")
# end


# @testset "sparsity" begin
#     include("sparsity.jl")
# end

# @testset "MNIST" begin
#     include("mnist.jl")
# end

# @testset "BPI2" begin # TODO change to MeanField
#     include("bpi2.jl")
# end

# if CUDA.functional()
#     @testset "CUDA" begin
#         include("cuda.jl")
#     end
# end
