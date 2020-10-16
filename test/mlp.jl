## COMMITTEE
for lay in [:tapex, :bpex]
    @time g, W, teacher, E = DeepMP.solve(α=0.2, K=[1001,7,1]
                , layers=[:tap,lay]
                , verbose=0
                ,r=.8,rstep=0.01, ry=0.3, seedξ=2, maxiters=500);
    @test E == 0
end

## ### COMMITTEE CONTINUOUS FIRST LAYER
##### COME MAI è così lento????
## @time g, W, teacher, E = DeepMP.solve(K=[301,5,1] ,layers=[:bpreal,:bpex]
##                    ,r=0.2,rstep=0.002, ry=0.2, altconv=true, altsolv=true, seedξ=1,
##                    maxiters=1000, plotinfo=0,β=Inf, α=2.);
## @test E == 0
##
## ## 3 LAYERS
for lay in [:tapex] #TODO  :bpex non ce la fa
    @time g, W, teacher, E = DeepMP.solve(α=0.2, K=[401,21,3,1]
                , layers=[:tap,lay,lay]
                , verbose=0
                ,r=.92,rstep=0.001, ry=0.0, seedξ=1,maxiters=300);
    @test E <= 1
end

@time g, W, teacher, E = DeepMP.solve(α=0.2, K=[401,21,3,1]
            , layers=[:tap,:bpex,:tapex]
            , verbose=0
            ,r=.95,rstep=0.001, seedξ=1,maxiters=200);
@test_broken E == 0
@test E < 5

@time g, W, teacher, E = DeepMP.solve(α=0.2, K=[401,21,3,1]
                        , layers=[:tap,:bp,:bpex]
                        , verbose=0
                        ,r=.9,rstep=0.002, seedξ=1,maxiters=1000);
@test E == 0

#### too slow
####  @time g, W, teacher, E = DeepMP.solve(α=0.2, K=[401,21,3,1]
####              , layers=[:tap,:bpex,:bpex]
####             ,r=.95,rstep=0.005, seedξ=1,maxiters=1000);
####

# @time g, W, teacher, E = DeepMP.solve(α=0.25, K=[301,21,11,3,1]
#                 , layers=[:tap,:tap,:tapex,:tapex]
#                 ,r=.9, rstep=0.001, ry=0.01, seedξ=1, maxiters=2000);
#
# @test E == 0

# IMPARATO TERZO LIVELLO!!!! (NON FUNZIONA PIU :(
# @time g, W, teacher, E = DeepMP.solve(α=0.2, K=[301,21,11,3,1]
#                 , layers=[:tap,:tap,:tapex,:bpex]
#                 ,r=.9, rstep=0.001, ry=0.01, seedξ=1, maxiters=500);

# @test E == 0


# @time g, W, teacher, E = DeepMP.solve(α=0.15, K=[301,21,11,3,1]
#                    , layers=[:tap,:tap,:tapex,:bpex]
#                    ,r=.9,rstep=0.0005,ry=0.01, seedξ=1,maxiters=2000);

##########################
