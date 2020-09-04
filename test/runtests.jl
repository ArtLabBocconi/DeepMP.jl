using DeepMP
using Test

@testset "Perceptron" begin
    include("perceptron.jl")
end

@testset "MLP" begin
    include("mlp.jl")
end
