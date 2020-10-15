@time g, W, teacher, E = DeepMP.solve(α=0.3, K=[201,5,1]
                        , layers=[:bp,:bpex]
                        , r=0.2, rstep=0.02, seedξ=2, maxiters=500, batchsize=1, density=0.5);

@assert E == 0