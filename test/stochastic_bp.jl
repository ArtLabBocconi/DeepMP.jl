@time g, W, teacher, E, it = DeepMP.solve(α=0.5, K=[1001, 1],
                    layers=[:bp], verbose=1,
                    r=0., rstep=0.0, ρ=1, 
                    seedx=2, 
                    maxiters=10, epochs=30, 
                    batchsize=1, density=1, altsolv=false, altconv=true);
@test E == 0

@time g, W, teacher, E, it = DeepMP.solve(α=0.3, K=[201,5,1],
                    layers=[:bp,:bpex],
                    r=0, rstep=0.0, 
                    ρ = 1,
                    altsolv=false, altconv=true,
                    maxiters=10, epochs=50,
                    seedx=2, batchsize=1, density=0.5);

@test E == 0