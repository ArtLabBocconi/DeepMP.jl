if DeepMP.F == Float64

    @testset "COMMITTEE" begin
        for lay in [:tapex, :bpex]
            @time g, W, teacher, E = DeepMP.solve(α=0.2, K=[1001,7,1]
                        , layers=[:tap,lay]
                        , verbose=0
                        , r=.9, rstep=0.01, seedx=2, maxiters=500);
            @test E == 0
        end
    end

    @testset "3 LAYERS" begin
        @test_broken begin
            @time g, W, teacher, E = DeepMP.solve(α=0.2, K=[401,21,3,1]
                        , layers=[:tap,:tapex,:tapex]
                        , verbose=0
                        ,r=.92,rstep=0.001, seedx=1,maxiters=300);
            
            @test_broken E == 0
            @test_broken E <= 100
        end
        
        @time g, W, teacher, E = DeepMP.solve(α=0.2, K=[401,21,3,1]
                    , layers=[:tap,:bpex,:tapex]
                    , verbose=0
                    ,r=.95,rstep=0.001, seedx=1,maxiters=200);

        @test E == 0
        @test E < 50

        @time g, W, teacher, E = DeepMP.solve(α=0.2, K=[401,21,3,1],
                                layers=[:tap,:bp,:bpex],
                                verbose=0,
                                r=.9,rstep=0.002, seedx=1,maxiters=800);
        @test E == 0
        
        #### too slow
        ####  @time g, W, teacher, E = DeepMP.solve(α=0.2, K=[401,21,3,1]
        ####              , layers=[:tap,:bpex,:bpex]
        ####             ,r=.95,rstep=0.005, seedx=1,maxiters=1000);
        ####
    end

    ## VERY SLOW
    # @time g, W, teacher, E = DeepMP.solve(α=0.25, K=[301,21,11,3,1]
    #                 , layers=[:tap,:tap,:tapex,:tapex]
    #                 ,r=.9, rstep=0.001, seedx=1, maxiters=2000);

    # @test_broken E == 0
    # @test_broken E < 100
    # @test E < 500

    @test_broken begin
        @time g, W, teacher, E = DeepMP.solve(α=0.2, K=[301,21,11,3,1],
                            layers=[:tap,:tap,:tapex,:bpex], verbose=0,
                            r=.9, rstep=0.001, seedx=1, maxiters=300);

        @test_broken E == 0
        @test_broken E < 10
        @test_broken E < 100
    end 
end
