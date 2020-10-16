module runexp

using Statistics, LinearAlgebra, Random, Printf

using DeepMP

function gen_error(w, wT; α=1.0, N=0, M=0)

    N == 0 && (N = length(w[1][1]))
    M == 0 && (M = round(Int, N * α))
    x = rand([-1.,1.], N, M)
    y = Int.(DeepMP.forward(wT, x)) |> vec
    ŷ = Int.(DeepMP.forward(w, x)) |> vec
    return sum(y .!= ŷ)    
end

function ts_overlap(w, wt)
    L = length(w)
    L > 1 && (L=L-1)
    Qts = 0.0
    for l = 1:L
        Q = 0.0
        K = length(w[l])
        for k = 1:K
            w0 = filter(x -> x != 0.0, w[l][k])
            wt0 = filter(x -> x != 0.0, wt[l][k])
            n0 = length(w0)
            @assert length(wt0) == n0
            # Q += mean(w0 .== wt0)
            Q += dot(w0, wt0) / n0
            # println("k=$k Q=$Q")
        end
        Q /= K
        Qts += Q
        # println("l=$l, K=$K Q=$Q Qts=$Qts")
    end
    return Qts / L
end

function runexpTS(;K::Vector{Int} = [501,5,1],
                  αrange::Union{Float64,Vector{Float64}} = 0.6,
                  layers = [:bpacc, :bpacc, :bpex],
                  seedξ::Int = -1,
                  seed::Int = -1,
                  r::Float64 = 0.9,
                  rstep::Float64 = 0.01,
                  y::Float64 = 0.0,
                  maxiters::Int = 1000,
                  epochs::Int = 1000,
                  batchsize::Int = 1,
                  altconv::Bool = false,
                  altsolv::Bool = false,
                  ϵ::Float64 = 1e-5,
                  ψ::Float64 = 0.5,
                  density::Union{Float64, Vector{Float64}} = 1.0,
                  use_teacher_weight_mask::Bool = true,
                  Mtest::Int = 1000,
                  verbose::Int=0,
                  plotinfo::Int=-1,
                  outfile::String = "")

    numW = length(K)==2 ? K[1]*K[2]  : sum(l->K[l]*K[l+1],1:length(K)-2)

    L = length(K)-1
    if isa(density, Number)
        density = fill(density, L)
        density[L] = 1
    end

    !isempty(outfile) && (f = open(outfile, "w"))

    for α in αrange
        g, w, wT, E = DeepMP.solve(α=α;    TS=true, 
                                           K=K,
                                           layers=layers,
                                           r=r,
                                           rstep=rstep,
                                           y=y,
                                           seedξ=seedξ,
                                           seed=seed,
                                           maxiters=maxiters,
                                           epochs=epochs,
                                           batchsize=batchsize,
                                           altconv=altconv,
                                           altsolv=altsolv,
                                           ϵ=ϵ,
                                           ψ=ψ,
                                           density=density,
                                           use_teacher_weight_mask=use_teacher_weight_mask,
                                           verbose=verbose,
                                           plotinfo=plotinfo);

        #Egen = gen_error(w, wT; αT=1.0, N=numW)
        Egen = gen_error(w, wT; M=Mtest) / Mtest
        Qts  = ts_overlap(w, wT)
        out  = @sprintf("α=%.2f, E=%i, Eg=%.3f, Qts=%.3f, ", α, E, Egen, Qts)
        outf = @sprintf("%f %i %f %f ", α, E, Egen, Qts)
        for l = 1:(L-1)
            q0, qab, R = DeepMP.compute_overlaps(g.layers[l+1])
            q0, q0_err, qab, qab_err = mean(q0), std(q0), mean(qab), std(qab)
    
            out  *= @sprintf("δ[%i]=%.2f, q0=%.2f±%.2f, qab=%.2f±%.2f ", 
                        l, density[l], q0, q0_err, qab, qab_err)
            outf *= @sprintf("%f %f %f %f %f ", density[l], q0, q0_err, qab, qab_err)
        end
        #!isempty(outfile) && println(f, "$outf")
        !isempty(outfile) && println(f, outf)
        !isempty(outfile) && flush(f)
        print("$out\n")
    end
    !isempty(outfile) && close(f)
end

const VecVecVec = Vector{Vector{Vector{<:Number}}}
function convert_weights(w::VecVecVec)
    w = [Matrix{Float32}(hcat(w[i]...)') for i = 1:length(w)]
    return w
end

function hamming_distance(w1, w2)
    w1 = convert_weights(w1)
    w2 = convert_weights(w2)
    N = sum(length, w1)
    d = 0.5 * (1.0 - (dot(w1, w2) / N))
    return d
end

function forward_sign(w, x)
    w = convert_weights(w)
    for i = 1:length(w)
        x = w[i] * x
        norm = size(w[1])[2]
        x = sign.(x ./ √norm)
    end
    return vec(x)
end


function runexpMLP(;K::Vector{Int} = [501,5,1],
                    ρrange::Union{Float64,Vector{Float64}} = 0.6,
                    layers = [:bpacc, :bpacc, :bpex],
                    seedξ::Int = -1,
                    seed::Int = -1,
                    w0 = [],
                    h0 = nothing,
                    xtrn = [], ytrn = [],
                    xtst = [], ytst = [],
                    r::Float64 = 0.0,
                    rstep::Float64 = 0.0,
                    y::Float64 = 0.0,
                    maxiters::Int = 100,
                    epochs::Int = 100,
                    batchsize::Int = 1,
                    altconv::Bool = false,
                    altsolv::Bool = false,
                    ϵ::Float64 = 1e-4,
                    ψ::Float64 = 0.0,
                    verbose::Int=0,
                    verbose_in::Int=0,
                    plotinfo::Int=-1,
                    outfile::String = "")

    numW = length(K)==2 ? K[1]*K[2]  : sum(l->K[l]*K[l+1],1:length(K)-2)

    !isempty(outfile) && (f = open(outfile, "w"))

    for ρ in ρrange
        g, w, wT, E, it = DeepMP.solve(xtrn, ytrn;
                                             K=K,
                                             layers=layers,
                                             h0=h0,
                                             ρ=ρ,
                                             r=r,
                                             rstep=rstep,
                                             y=y,
                                             seed=seed,
                                             maxiters=maxiters,
                                             epochs=epochs,
                                             batchsize=batchsize,
                                             altconv=altconv,
                                             altsolv=altsolv,
                                             ϵ=ϵ,
                                             ψ=ψ,
                                             density=1,
                                             verbose=verbose,
                                             verbose_in=verbose_in,
                                             plotinfo=plotinfo);

        Egen  = mean(vec(DeepMP.forward(w, xtst)) .== ytst)
        EgenT = mean(vec(DeepMP.forward(h0, xtst)) .== ytst)

        # dist = hamming_distance(w, h0)
        R = ts_overlap(w, h0)
        out  = @sprintf("ρ=%.2f, E=%i, Eg=%.3f, EgT=%.3f, R=%.3f, it=%i ", ρ, E, Egen, EgenT, R, it)
        outf = @sprintf("%f %i %f %f %i ", ρ, E, Egen, R, it)
        L = length(K)-1
        for l = 1:(L-1)
            q0, qab, R = DeepMP.compute_overlaps(g.layers[l+1])
            q0, q0_err, qab, qab_err = mean(q0), std(q0), mean(qab), std(qab)
            out  *= @sprintf("l=%i, q0=%.2f±%.2f, qab=%.2f±%.2f ", l, q0, q0_err, qab, qab_err)
            outf *= @sprintf("%f %f %f %f ", q0, q0_err, qab, qab_err)
        end
        #!isempty(outfile) && println(f, "$outf")
        !isempty(outfile) && println(f, outf)
        !isempty(outfile) && flush(f)
        print("$out\n")
    end
    !isempty(outfile) && close(f)
end


end # module
