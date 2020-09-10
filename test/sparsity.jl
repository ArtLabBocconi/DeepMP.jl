y = Int.(readdlm("../fmnist/seed7/Y.txt")) |> vec
X = readdlm("../fmnist/seed7/X.txt")

@time g, W, E, stab = DeepMP.solve(X,y, K=[784,5, 1]
                   , layers=[:bpacc, :bpex]
                   ,r=.9,rstep=0.01, seed=2,maxiters=500, density=0.5);


@test E == 0