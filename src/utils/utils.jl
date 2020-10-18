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


#TODO use vector of matrix teacher
function rand_teacher(K::Vector{Int}; density=1.)
    L = length(K)-1
    @assert K[L+1] == 1
    density = process_density(density, L)
    
    T = Float64
    W = Vector{Vector{Vector{T}}}()
    for l=1:L
        push!(W, [rand(T[-1,1], K[l]) for k=1:K[l+1]])
        for k in 1:K[l+1]
            W[l][k] .*= [rand() < density[l] ? 1 : 0 for i=1:K[l]]
        end
    end
    if L > 1
        W[L][1] .= 1
    end
    return W
end

