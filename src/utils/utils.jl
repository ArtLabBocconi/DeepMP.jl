function printvec(q::Vector{Float64}, head = "")
    print(head)
    if length(q) < 10
        for e in q
            @printf("%.6f ", e)
        end
    else
        @printf("mean:%.6f std:%.6f", mean(q), std(q))
    end
    println()
end

function meanoverlap(ξ::Matrix)
    N, M =size(ξ)
    q = 0.
    for a=1:M
        for b=a+1:M
            q += dot(ξ[:,a],ξ[:,b])
        end
    end
    return q / N / (0.5*M*(M-1))
end

function create_minibatches(M::Int, batchsize::Int)
    batchsize > M && (@warn "Batchsize ($batchsize) > M ($M)")

    num_batches = M ÷ batchsize + ((M % batchsize) == 0 ? 0 : 1)
    patt_perm = randperm(M)
    minibatches = Vector{Int}[]
    for b = 1:num_batches
        first = (batchsize*(b-1)+1)
        last = (batchsize * b > M ? M : batchsize * b)
        batch = patt_perm[first:last]
        push!(minibatches, batch)
    end
    @assert length(minibatches) == num_batches
    return minibatches
end
function create_minibatch(M::Int, batchsize::Int)
    batchsize > M && (@warn "Batchsize ($batchsize) > M ($M)")
    last = (batchsize > M ? M : batchsize)
    batch = randperm(M)[1:last]
    return batch
end
