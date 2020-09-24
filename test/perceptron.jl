##### PERCEPTRON
for lay in [:tap,:bp]
    @time g, W, teacher, E, stab = DeepMP.solve(α=0.7, K=[1001,1]
    , layers=[lay], verbose=0
    ,r=.2,rstep=0.01, seedξ=1,maxiters=500);
    @test E == 0
end

for lay in [:bpi, :ms]
    @time g, W, teacher, E, stab = DeepMP.solve(α=0.5, K=[201,1]
                , layers=[lay]
                ,r=1.,rstep=0.0, seedξ=1,maxiters=1000);
    @test E == 0
end

for lay in [:tapex,:bpex]
    @time g, W, teacher, E, stab = DeepMP.solve(α=0.7, K=[101,1]
                , layers=[lay], verbose=0
                , r=.3, rstep=0.002, seedξ=1,maxiters=500);
    @test E == 0
end