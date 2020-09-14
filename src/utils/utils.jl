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
