mutable struct ReinfParams
    r::Float64         # reinforcement (γ for focusing) for W variables
    rstep::Float64
    ry::Float64         # reinforcement for Y variables
    rystep::Float64
    y::Float64          # parameter for FocusingBP
    ψ::Float64          # damping parameter
    wait_count::Int
    ReinfParams(r=0., rstep=0., ry=0., rystep=0., y=0, ψ=0.) = new(r, rstep, ry, rystep, y, ψ, 0)
end

function update_reinforcement!(reinfpar::ReinfParams)
    if reinfpar.wait_count < 10
        reinfpar.wait_count += 1
    else
        if reinfpar.y <= 0
            # reinforcement update
            reinfpar.r = 1 - (1-reinfpar.r) * (1-reinfpar.rstep)
            reinfpar.ry = 1 - (1-reinfpar.ry) * (1-reinfpar.rystep)
        else
            # focusing update
            reinfpar.r *= (1.0 + reinfpar.rstep)
            #@assert false #TODO
        end
    end
end
