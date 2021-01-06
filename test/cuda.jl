using Test
using CUDA
CUDA.allowscalar(false)

usecuda = true
verbose = 0

@testset "Perceptron" begin 
for lay in [:bp]
    @time g, W, teacher, E = DeepMP.solve(; α=0.5, K=[1001,1],
            layers=[lay], verbose, usecuda,
            r=.2,rstep=0.01, seedx=1,maxiters=200);

    @test E == 0
end

### teacher-student
for lay in [:bp]
    @time g, W, teacher, E = DeepMP.solve(; α=0.5, K=[1001,1],
            layers=[lay], verbose, TS=true, usecuda,
            r=.2,rstep=0.01, seedx=1,maxiters=200);
            
    @test E == 0
end

@time g, W, teacher, E, it = DeepMP.solve(; α=0.5, K=[1001, 1],
                    layers=[:bp], verbose,
                    r=0., rstep=0.0, ρ=1, 
                    seedx=2, usecuda,
                    maxiters=10, epochs=30, 
                    batchsize=1, density=1, altsolv=false, altconv=true);
@test E == 0

end