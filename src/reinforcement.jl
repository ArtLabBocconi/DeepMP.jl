mutable struct ReinfParams
    r::Union{Float32, Vector{Float64}}   # reinforcement (γ for focusing) for W variables
    rstep::Float32
    y::Float32                           # parameter for FocusingBP
    ψ::Union{Float32, Vector{Float64}}   # damping parameter
    wait_count::Int
    l::Int
end

ReinfParams(r=0., rstep=0., y=0, ψ=0.) = ReinfParams(r, rstep, y, ψ, 0, 0)


function update_reinforcement!(reinfpar::ReinfParams)
    if reinfpar.wait_count < 10
        reinfpar.wait_count += 1
    else
        if reinfpar.y <= 0
            # reinforcement update
            reinfpar.r .= 1 .- (1 .- reinfpar.r) .* (1 - reinfpar.rstep)
        else
            # focusing update
            reinfpar.r .*= (1.0 + reinfpar.rstep)
            #@assert false #TODO
        end
    end
end
