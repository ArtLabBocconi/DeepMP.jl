##### PERCEPTRON
for lay in [:tap,:bp]
    local g, W, teacher, E

    @time g, W, teacher, E = DeepMP.solve(α=0.7, K=[1001,1]
            , layers=[lay], verbose=0
            ,r=.2,rstep=0.01, seedx=1,maxiters=500);
    
    @test E == 0
end

# teacher-student
for lay in [:tap,:bp]
    local g, W, teacher, E

    @time g, W, teacher, E = DeepMP.solve(α=0.7, K=[1001,1]
            , layers=[lay], verbose=0, TS=true
            ,r=.2,rstep=0.01, seedx=1,maxiters=500);
            
    @test E == 0
end

if DeepMP.F == Float64
    @time g, W, teacher, E = DeepMP.solve(α=0.5, K=[201,1]
                    , layers=[:bpi], verbose=0
                    , altsolv=true, altconv=false
                    ,r=1.,rstep=0.0, seedx=1,maxiters=1000);
    @test E == 0
end

for lay in [:tapex,:bpex]
    local g, W, teacher, E

    @time g, W, teacher, E = DeepMP.solve(α=0.7, K=[101,1]
                , layers=[lay], verbose=0
                , r=.3, rstep=0.002, seedx=1,maxiters=500);
    @test E == 0
end