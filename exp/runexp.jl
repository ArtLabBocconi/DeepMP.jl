module runexp

using Statistics, LinearAlgebra, Random, Printf

include("../src/DeepMP.jl")

function gen_error(w, wT; α=1.0, N=0, M=0)

    N == 0 && (N = length(w[1][1]))
    M == 0 && (M = round(Int, N * α))
    ξ = rand([-1.,1.], length(w[1][1]), M)
    σ = Int[DeepMP.forward(wT, ξ[:, a])[1][1] for a=1:M]
    @assert (any(i -> i == 0, σ) == false)

    Egen =  sum(Int[DeepMP.forward(w, ξ[:, a])[1][1] != σ[a] for a=1:(size(ξ)[2])])
    return Egen
end

function overlaps(g, l)
    K = g.K
    L = length(K)-1
    @assert (l <= L)
    N = K[1]
    layer = g.layers[2:end-1][l]

    mask = layer.weight_mask

    q0 = Float64[]
    for k=1:K[l+1]
        norm = K[l] * mean(mask[k])
        #push!(q0, dot(layers[l].allm[k], layers[l].allm[k])/K[l])
        push!(q0, dot(layer.allm[k], layer.allm[k]) / norm)
    end
    qab = Float64[]
    for k=1:K[l+1]
        for p=k+1:K[l+1]
            norm = sqrt(q0[k] * q0[p]) * K[l] * sqrt(mean(mask[k]) * mean(mask[p]))
            #push!(qWαβ, dot(layers[l].allm[k],layers[l].allm[p]) / sqrt(q0[k]*q0[p])/K[l])
            push!(qab, dot(layer.allm[k],layer.allm[p]) / norm)
        end
    end

    return mean(q0), std(q0), mean(qab), std(qab)
end

function runexpTS(;K::Vector{Int} = [501,5,1],
                  αrange::Union{Float64,Vector{Float64}} = 0.6,
                  layers = [:bpacc, :bpacc, :bpex],
                  seedξ::Int = -1,
                  seed::Int = -1,
                  r::Float64 = 0.9,
                  rstep::Float64 = 0.01,
                  maxiters::Int = 1000,
                  ψ::Float64 = 0.5,
                  density::Union{Int, Vector{Int}} = 1,
                  Mtest::Int = 1000,
                  verbose::Int=0,
                  plotinfo::Int=-1,
                  outfile::String = "")

    @assert length(K) < 5 # for now max 2 hidden layers
    numW = length(K)==2 ? K[1]*K[2]  : sum(l->K[l]*K[l+1],1:length(K)-2)

    L = length(K)-1
    if isa(density, Number)
        density = fill(density, L)
        density[L] = 1
    end

    !isempty(outfile) && (f = open(outfile, "w"))

    for α in αrange
        g, w, wT, E, stab = DeepMP.solveTS(α=α;
                                           K=K,
                                           layers=layers,
                                           r=r,
                                           rstep=rstep,
                                           seedξ=seedξ,
                                           seed=seed,
                                           maxiters=maxiters,
                                           ψ=ψ,
                                           density=density,
                                           verbose=verbose,
                                           plotinfo=plotinfo);


        #Egen = gen_error(w, wT; αT=1.0, N=numW)
        Egen = gen_error(w, wT; M=Mtest) / Mtest
        out  = @sprintf("α=%.2f, E=%i, Eg=%.3f, ", α, E, Egen)
        outf = @sprintf("%f %i %f ", α, E, Egen)
        for l = 1:(L-1)
            q0, q0_err, qab, qab_err = overlaps(g, l)
            out  *= @sprintf("δ[%i]=%.2f, q0=%.2f±%.2f, qab=%.2f±%.2f ", l, density[l], q0, q0_err, qab, qab_err)
            outf *= @sprintf("%f %f %f %f %f ", density[l], q0, q0_err, qab, qab_err)
        end
        !isempty(outfile) && println(f, "$outf")
        print("$out\n")
    end
    !isempty(outfile) && close(f)
end



end # module
