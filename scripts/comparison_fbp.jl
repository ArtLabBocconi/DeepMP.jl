using DeepMP
using BinaryCommitteeMachineFBP

tfbp = @timed  focusingBP(321, 5, 0.3, 
                        randfact=0.1, seed=135, 
                        max_steps=10, max_iters=1, 
                        quiet=true,
                        accuracy1=:accurate, # :none, :accurate, :exact
                        accuracy2=:exact, # :none, :accurate, :exact
                        messfmt = :plain,
                        damping=0.0);

errs, messages, patterns = tfbp.value
@show tfbp.time

xtrain = hcat(patterns.X...)
ytrain = fill(1, size(xtrain, 2))

tdeep = @timed DeepMP.solve(
                xtrain, ytrain;
                K = [321, 5, 1],
                layers = [:bp, :bpex],
                maxiters=10,
                r = 0., rstep=0.,
                batchsize=-1, #epochs = 50,
                altsolv=false, altconv=false, 
                freezetop=true,
                verbose=0,
                ρ = 1, 
                )

@show tdeep.time
