@testset "sign gpu" begin
    act = DeepMP.channel(:sign)
    x = rand(10) |> cu
    @test act(x) |> Array == ones(10) 
    if x isa CuArray
        @test_broken act.(x) |> Array == ones(10)
    end
end
