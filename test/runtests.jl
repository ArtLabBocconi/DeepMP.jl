using DeepMP
using Test, DelimitedFiles, Random, Statistics

@testset "Perceptron" begin
    include("perceptron.jl")
end

@testset "Stochastic BP" begin
    include("stochastic_bp.jl")
end

@testset "BPI_MLP" begin
    include("bpi_mlp.jl")
end

@testset "MLP" begin
    include("mlp.jl")
end

@testset "sparsity" begin
    include("sparsity.jl")
end

@testset "MNIST" begin
    include("mnist.jl")
end

@testset "BPI2" begin
    include("bpi2.jl")
end
