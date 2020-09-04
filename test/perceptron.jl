##### PERCEPTRON
for lay in [:tap,:bp]
    @time g, W, E, stab = DeepMP.solve(α=0.7, K=[1001,1]
    , layers=[lay]
    ,r=.2,rstep=0.01, seedξ=1,maxiters=500);
    @test E == 0
end

@time g, W, E, stab = DeepMP.solve(α=0.5, K=[201,1]
            , layers=[:ms]
            ,r=1.,rstep=0.0, seedξ=1,maxiters=1000);
@test E == 0

for lay in [:tapex,:bpex]
    @time g, W, E, stab = DeepMP.solve(α=0.7, K=[101,1]
                , layers=[lay]
                ,r=.3,rstep=0.002, seedξ=1,maxiters=500);
    @test E == 0
end