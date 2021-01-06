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


function rand_teacher(K::Vector{Int}; density=1.)
    L = length(K)-1
    @assert K[L+1] == 1
    density = process_density(density, L)
    
    W = Vector{Matrix{F}}()
    for l=1:L
        w = rand(F[-1,1], K[l+1], K[l])
        w .*= rand!(similar(w)) .< density[l]
        push!(W, w)
    end
    if L > 1
        W[L] .= 1
    end
    return W
end

