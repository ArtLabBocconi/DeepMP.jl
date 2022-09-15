module LocalEnergy

using Pkg
Pkg.activate("../")
Pkg.instantiate()

using DeepMP
using Statistics, LinearAlgebra, Random
using DelimitedFiles, Printf

⊗(w1, w2) = ((x, y) -> x .* y).(w1, w2)

function hamming_distance(w1, w2)
    H = length(w1)
    @assert length(w2) == H
    wvec1 = vcat([vec(w1[h]) for h in 1:H]...)
    wvec2 = vcat([vec(w2[h]) for h in 1:H]...)
    return mean(wvec1 .!= wvec2)
end

function create_noise(w, ϵ)
    noise = Array{Any}[]
    H = length(w)
    for h = 1:H
        push!(noise, 2.0 .* (rand(size(w[h])...) .> ϵ) .- 1.0)
    end
    return noise
end

signB(x::T) where {T} = sign(x + 1f-10)

# Binary Classification
function forward(w, x)
    L = length(w)
    for l = 1:L
        x = signB.(w[l] * x)
    end
    return vec(x)
end
# Multiclass
function forward_mc(w, x)
    L = length(w)
    for l = 1:(L-1)
        x = signB.(w[l] * x)
    end
    ŷ = argmax(w[L] * x, dims=1)
    return vec(getindex.(ŷ, 1))
end

function errors(w, x, y)
    if maximum(y) > 1
        return mean(forward_mc(w, x) .!= y)
    else
        return mean(forward(w, x) .!= y)
    end
end

# function energy_profile(g::FactorGraph, x, y; kws...)
#     energy_profile(DeepMP.getW(g), x, y; kws...)
# end

function energy_profile(w::Vector{Matrix{Float64}}, x, y;
                        seed::Int=-1,
                        max_iters::Int=100,
                        ϵrange=LinRange(0.0, 0.3, 30),
                        verbose=true,
                        outfile="")

    isa(outfile, String) && !isempty(outfile) && (f = open(outfile, "w"))
    seed > 0 && Random.seed!(seed)

    res = []
    for ϵ in ϵrange
        errs = []
        dists = []
        for i = 1:max_iters
            noise = create_noise(w, ϵ)
            err = errors(⊗(w, noise), x, y)
            dist = hamming_distance(w, ⊗(w, noise))
            push!(errs, err)
            push!(dists, dist)
        end
        err = mean(errs)
        Δerr = std(errs) / sqrt(length(errs)-1.0)
        dist = mean(dists)
        Δdist = std(dists) / sqrt(length(dists)-1.0)
        verbose && @printf("ϵ=%.3f, d=%.3f±%.3f, err=%.3f±%.3f\n", ϵ, dist, Δdist, err, Δerr)
        push!(res, [ϵ dist Δdist err Δerr])
    end
    if isa(outfile, String) && !isempty(outfile)
        verbose && @info "Writing on $outfile"
        writedlm(outfile, vcat(res))
    end

    isa(outfile, String) && !isempty(outfile) && close(f)

    # (length(ϵrange) == 1) && return res[1][4] # i.e. the error at ϵ

end # energy_profile

end # module
