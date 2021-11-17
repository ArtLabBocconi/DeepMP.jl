#### ACT RELU   ###############

## type hierarchy for input and output channels


@kwdef mutable struct ActReLU <: AbstractChannel
    name::Symbol
end

(ch::ActReLU)(x) =  max(0, x)

# intermediate layer
function ∂ω_ϕout(ch::ActReLU, A, B, ω, V)
    c = ω
    d = V^2
    E1, H1, EH2, Z = compute_relu_exps(A, B, c, d)
    
    if A > 1e3 && B > 1e3
        g = updg_inf_relu(B/A, c, d)
    elseif EH2 > 1e5*H1
     # print("!")
        g = (B-A*c)/(1 + A*d)
    else
        g = (-A*√d*E1/(1+A*d) + EH2*(B-A*c)/(1 + A*d)) / Z
    end
    return g
end

function ∂²ω_ϕout(ch::ActReLU, A, B, ω, V)
    c = ω
    d = V^2

    E1, H1, EH2, Z = compute_relu_exps(A, B, c, d)
    g = ∂ω_ϕout(ch, A, B, ω, V)
    
    if A > 1e3 && B > 1e3
        ∂g = upd∂g_inf_relu(B/A, c, d)
    elseif EH2 > 1e5*H1
        # print("!")
        ∂g = -(A^2*(d-c^2)+2A*B*c+A-B^2)/(1 + A*d)^2 - g^2
    else
        ∂g = ((A^2*c*d+B)*E1/(√d*(1+A*d)^2)
        - EH2*(A^2*(d-c^2)+2A*B*c+A-B^2)/(1 + A*d)^2) / Z - g^2
    end
    -∂g
end

function ∂B_ϕout(ch::ActReLU, Atot, Btot)
    Btot / Atot
end

function ∂B²_ϕout(ch::ActReLU, Atot, Btot)
    1 / Atot
end

function Bout_ϕout(ch::ActReLU, ω, V)
    c = ω
    d = V^2

    E1, H1, EH2, Z = compute_relu_exps(0, 0, c, d)

    H1 = @. H(c/√d)
    E1 = @. exp(-c^2/(2d)) / √(2π)

    a = @. (B^2*d+2*B*c-A*c^2)/(2*(1+A*d))
    b = @. -(c+B*d)/√(d*(1+A*d))
    # H2 = @. H(-(c+B*d)/√(d*(1+A*d)))
    # E2 = @. exp(a) / √(1 + A*d)
    # EH2 = E2 .* H2
    EH2 = @. exp(a + logH(b)) / √(1 + A*d) 
    Z = @. H1 + EH2
    @assert all(isfinite.(H1))
    @assert all(isfinite.(EH2))
    @assert all(isfinite.(E1))
    # @assert all(isfinite.(E2))
    @assert all(isfinite.(Z))
    
    @. hnew = updh_relu(A, B, c, d, H1, E1, EH2)

    Δĥ = mean(abs.(hnew .- ĥ))
    ĥ = hnew

    @. σ = updσ_relu(A, B, c, d, H1, E1, EH2, ĥ)
    
end


function compute_relu_exps(A, B, c, d)
    a =  (B^2*d+2*B*c-A*c^2)/(2*(1+A*d))
    b =  -(c+B*d)/√(d*(1+A*d))

    H1 = H(c/√d)
    H2 = H(b)
    
    E1 = exp(-c^2/(2d)) / √(2π)
    E2 = exp(a) / √(1 + A*d)

    EH2 = exp(a + logH(b)) / √(1 + A*d) 
    Z = H1 + EH2

    E1, H1, EH2, Z
end


updg_inf_relu(B, c, d) = B > 0 ? (B - c) / d : -GH(c/√d) / √d

function upd∂g_inf_relu(B, c, d)
    if B > 0
        -1 / d
    else
        g = -GH(c / √d) / √d
        -c/d* g - g^2
    end
end


# function updg_relu(A, B, c, d, H1, H2, E1, E2)
#     if A > 1e3 && B > 1e3
#         g = updg_inf_relu(B/A, c, d)
#     elseif E2*H2 > 1e5*H1
#      # print("!")
#         g = (B-A*c)/(1 + A*d)
#     else
#         Z = H1 + H2 * E2
#         g = (-A*√d*E1/(1+A*d) + E2*H2*(B-A*c)/(1 + A*d)) / Z
#     end
#     return g
# end

# function upd∂g_relu(A, B, c, d, H1, H2, E1, E2, g)
#     if A > 1e3 && B > 1e3
#         ∂g = upd∂g_inf_relu(B/A, c, d)
#     elseif E2*H2 > 1e5*H1
#         # print("!")
#         ∂g = -(A^2*(d-c^2)+2A*B*c+A-B^2)/(1 + A*d)^2 - g^2
#     else
#         Z = H1 + H2 * E2
#         ∂g = ((A^2*c*d+B)*E1/(√d*(1+A*d)^2)
#         - E2*H2*(A^2*(d-c^2)+2A*B*c+A-B^2)/(1 + A*d)^2) / Z - g^2
#     end
#     return ∂g
# end

# function update_gdg!(::ActReLU, g, ∂g, A, B, c, d)
#     if any(A .== Inf)
#         @assert all(A .== Inf)
#         @assert all(B .>= 0)
#         @. g =  updg_inf_relu(B, c, d)
#         @. ∂g = upd∂g_inf_relu(B, c, d)
#     else
#         H1 = @. H(c/√d)
#         H2 = @. H(-(c+B*d)/√(d*(1+A*d)))
#         E1 = @. exp(-c^2/(2d)) / √(2π)
#         E2 = @. exp((B^2*d+2*B*c-A*c^2)/(2*(1+A*d))) / √(1 + A*d)

#         @assert all(isfinite.(H1))
#         @assert all(isfinite.(H2))
#         # @assert all(isfinite.(E2)) "A=$A \n  B=$B"

#         @. g = updg_relu(A, B, c, d, H1, H2, E1, E2)

#         @. ∂g = upd∂g_relu(A, B, c, d, H1, H2, E1, E2, g)
#     end
#     if any(isnan.(g))
#         println("!")
#         I = find(isnan, g)
#         @show H1[I] H2[I] E1[I] E2[I] A[I] B[I]
#     end

#     @assert all(isfinite.(g))
#     @assert all(isfinite.(∂g))
# end


function updh_relu(A, B, c, d, H1, E1, EH2)
    if EH2 > 1e5*H1
        hnew = (B*d+c)/(1 + A*d)
    else
        Z = H1 + EH2
        hnew = (√d*E1/(1+A*d) + EH2*(B*d+c)/(1 + A*d)) / Z
    end
    return hnew
end
function updσ_relu(A, B, c, d, H1, E1, EH2, ĥ)
    if EH2 > 1e5*H1
        σ =  (d*((A+B^2)*d + 1) + 2B*c*d+c^2)/(1 + A*d)^2 - ĥ^2
    else
        Z = H1 + EH2
        σ = (√d*(B*d+c)*E1/(1+A*d)^2
                + EH2*(d*((A+B^2)*d + 1) + 2B*c*d+c^2)/(1 + A*d)^2) / Z - ĥ^2
    end
    return σ
end

function update_ĥσ!(ĥ, hnew, σ, A, B, c, d, act::ActReLU, updmask)
    @assert all(isfinite.(A))
    @assert all(isfinite.(B))
    @assert all(isfinite.(c))
    @assert all(isfinite.(d))

    H1 = @. H(c/√d)
    E1 = @. exp(-c^2/(2d)) / √(2π)

    a = @. (B^2*d+2*B*c-A*c^2)/(2*(1+A*d))
    b = @. -(c+B*d)/√(d*(1+A*d))
    # H2 = @. H(-(c+B*d)/√(d*(1+A*d)))
    # E2 = @. exp(a) / √(1 + A*d)
    # EH2 = E2 .* H2
    EH2 = @. exp(a + logH(b)) / √(1 + A*d) 
    Z = @. H1 + EH2
    @assert all(isfinite.(H1))
    @assert all(isfinite.(EH2))
    @assert all(isfinite.(E1))
    # @assert all(isfinite.(E2))
    @assert all(isfinite.(Z))
    
    @. hnew = updh_relu(A, B, c, d, H1, E1, EH2)

    Δĥ = mean(abs.(hnew .- ĥ))
    @. ĥ = hnew

    @. σ = updσ_relu(A, B, c, d, H1, E1, EH2, ĥ)

    @assert all(isfinite.(ĥ))
    @assert all(isfinite.(σ))

    return Δĥ
end
