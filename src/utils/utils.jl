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

function mags_symmetry(g, K_list)
    N, K = K_list[1], K_list[2]
    overlaps = Matrix(1.0I, K, K)
    wtemp = getW(g)
    #@show size(wtemp[1])
    #error()
    for k1 = 1:K, k2 = k1+1:K
        s = 0.0
        for i = 1:N
            s += (wtemp[1][k1, i] == wtemp[1][k2, i])
        end
        s /= N
        overlaps[k1, k2] = s
        overlaps[k2, k1] = s
    end
    return overlaps
end
