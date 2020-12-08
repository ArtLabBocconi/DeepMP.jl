using Test

##### PERCEPTRON
for lay in [:bp]
    @time g, W, teacher, E = DeepMP.solve(α=0.5, K=[1001,1],
            layers=[lay], verbose=1, usecuda=true,
            r=.2,rstep=0.01, seedx=1,maxiters=200);

    @test E == 0
end

teacher-student
for lay in [:bp]
    @time g, W, teacher, E = DeepMP.solve(α=0.5, K=[1001,1],
            layers=[lay], verbose=1, TS=true, usecuda=true,
            r=.2,rstep=0.01, seedx=1,maxiters=200);
            
    @test E == 0
end

@time g, W, teacher, E, it = DeepMP.solve(α=0.5, K=[1001, 1],
                    layers=[:bp], verbose=1,
                    r=0., rstep=0.0, ρ=1, 
                    seedx=2, usecuda=true,
                    maxiters=10, epochs=30, 
                    batchsize=1, density=1, altsolv=false, altconv=true);
@test E == 0
