module bpmlp

using Random, Statistics, DelimitedFiles, Printf

include("../src/DeepMP.jl")
include("../../binary-pruning/src/binmlp.jl")
include("../../binary-pruning/src/bindata.jl")

function convert_weights(w)
    w = [Matrix{Float32}(hcat(w[i]...)') for i = 1:length(w)]
    return w
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

function runexp(;
                seed=-1,
                h=[15, 15],
                dataset="cifar10",
                cl1=1,
                cl2=2,
                numex=500,
                reduce_input=false,
                outfile="out.dat",
                lr = 0.01,
                epochs=50,
                opt = "SGD",
                density = 1.0)

    f = open(outfile, "w")

    Wsgd = binmlp.main(seed=seed,
                       seed_data=23,
                       epochs=epochs,
                       h=h,
                       dataset=dataset,
                       cl1=cl1,
                       cl2=cl2,
                       B=100,
                       dim_train=numex,
                       reduce_input=reduce_input,
                       comp_loss=false,
                       opt=opt,
                       lr=lr,
                       outfile=nothing,
                       mask_fwd=true,
                       δ=density);

    xtrn, ytrn, xtst, ytst = bindata.getdata(dataset;
                                             seed=23,
                                             cl1=cl1, cl2=cl2,
                                             numex=numex,
                                             reduce_input=reduce_input,
                                             normalize_data=false,
                                             edge=:median, bintype=:default);


    wsgd = []
    for l = 1:length(Wsgd)
        push!(wsgd, [Wsgd[l][i,:] for i = 1:size(Wsgd[l])[1]])
    end
    K = [size(xtrn)[1], h..., 1]
    layers = [:bpacc for _ = 1:length(K)-1]

    g, wbp, wt, E, stab, it = DeepMP.solve(Array{Float64}(xtrn), ytrn,
                                           seed=seed,
                                           K=K,
                                           layers=layers,
                                           h0=nothing,
                                           teacher=wsgd,
                                           epochs=epochs,
                                           maxiters=10,
                                           ψ=0.0,
                                           density=density,
                                           batchsize=1,
                                           ϵ=1e-4,
                                           altsolv=true, altconv=true,
                                           verbose_in=0,
                                           use_teacher_weight_mask=true,
                                           xtest=xtst, ytest=ytst);

    trnacc_sgd  = mean(forward_sign(wsgd, xtrn) .== ytrn)
    trnacc_bp   = mean(forward_sign(wbp, xtrn) .== ytrn)
    tstacc_sgd  = mean(forward_sign(wsgd, xtst) .== ytst)
    tstacc_bp   = mean(forward_sign(wbp, xtst) .== ytst)

    out  = @sprintf("trnacc_sgd=%g, trnacc_bp=%g, tstacc_sgd=%g, tstacc_bp=%g", trnacc_sgd, trnacc_bp, tstacc_sgd, tstacc_bp)
    outf = @sprintf("%g %g %g %g", trnacc_sgd, trnacc_bp, tstacc_sgd, tstacc_bp)
    print("$out\n")
    println(f, outf)
    flush(f)
    !isempty(outfile) && close(f)

end

end # module
