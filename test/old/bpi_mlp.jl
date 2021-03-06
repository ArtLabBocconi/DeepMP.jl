## COMMITTEE

if DeepMP.F == Float64
    @time g, W, teacher, E = DeepMP.solve(α=0.2, K=[201,7,1]
                    , layers=[:bpi, :bpex]
                    , verbose=0
                    , r=.8, rstep=0.01, seedx=2, maxiters=500);
    @test E == 0

    for freezetop in [true, false], lay1 in [:bp, :bpacc, :bpi]

        @time g, W, teacher, E = DeepMP.solve(; α=0.2, K=[51,11,1],
                            maxiters=300, seedx=2,
                            r = 0.9, rstep=0.01, verbose=0,
                            altsolv =true, altconv=true, freezetop,
                            layers=[lay1,:bpi]);
        @test E == 0
    end


    ## 2 LAYERS

    @time g, W, teacher, E = DeepMP.solve(α=0.2, K=[201,11,3,1]
            , layers=[:bpi, :bpacc, :bpex]
            , verbose=0
            , r=0.9, rstep=0.01, seedx=1, maxiters=500);

    @test E == 0

    @test_broken begin
        @time g, W, teacher, E = DeepMP.solve(α=0.2, K=[201,11,3,1]
            , layers=[:bpi, :bpi, :bpex]
            , verbose=0
            , r=0.9, rstep=0.01, seedx=1, maxiters=500);
            
        @test_broken E == 0
        @test_broken E <= 10
        #@test_broken E <= 70
        @test E <= 70
        @test E < 90
        true
    end
    
    @time g, W, teacher, E = DeepMP.solve(α=0.2, K=[401,21,3,1],
                      layers=[:tap,:bpex,:tapex], verbose=0,
                      r=.95,rstep=0.001, seedx=1,maxiters=200);
    @test E == 0


    @time g, W, teacher, E = DeepMP.solve(α=0.2, K=[401,21,3,1],
                          layers=[:tap,:bp,:bpex], verbose=0,
                          r=.9, rstep=0.002, seedx=1,maxiters=200);

    @test_broken E == 0
    @test E <=20
end
