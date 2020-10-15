## COMMITTEE

@time g, W, teacher, E = DeepMP.solve(α=0.2, K=[201,7,1]
                , layers=[:bpi, :bpex]
                , r=.8, rstep=0.01, ry=0., seedξ=2, maxiters=500);
@test E == 0

## 2 LAYERS

@time g, W, teacher, E = DeepMP.solve(α=0.2, K=[201,11,3,1]
           , layers=[:bpi, :bpacc, :bpex]
           , r=0.9, rstep=0.01, ry=0.0, seedξ=1, maxiters=300);
@test E == 0

@time g, W, teacher, E = DeepMP.solve(α=0.2, K=[201,11,3,1]
           , layers=[:bpi, :bpi, :bpex]
           , r=0.9, rstep=0.01, ry=0.0, seedξ=1, maxiters=300);
@test E == 0

# for lay in [:tapex] #TODO  :bpex non ce la fa
# end

# @time g, W, teacher, E = DeepMP.solve(α=0.2, K=[401,21,3,1]
#             , layers=[:tap,:bpex,:tapex]
#             ,r=.95,rstep=0.001, seedξ=1,maxiters=200);
# @test E == 0

# @time g, W, teacher, E = DeepMP.solve(α=0.2, K=[401,21,3,1]
#                    , layers=[:tap,:bp,:bpex]
#                    ,r=.9,rstep=0.002, seedξ=1,maxiters=1000);
# @test E == 0
