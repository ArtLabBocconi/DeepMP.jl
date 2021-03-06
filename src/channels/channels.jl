##### CHANNELS  ##################
abstract type AbstractChannel end

∂ω_ϕout(ch::AbstractChannel, y, ω, V) = deriv(ω->ϕout(ch, y, ω, V), ω)
∂²ω_ϕout(ch::AbstractChannel, y, ω, V) = deriv(ω->∂ω_ϕout(ch, y, ω, V), ω)
∂B_ϕin(ch::AbstractChannel, B, A) = deriv(B->ϕin(ch, B, A), B)
∂²B_ϕin(ch::AbstractChannel, B, A) = deriv(B->∂B_ϕin(ch, B, A), B)

function compute_g(act::AbstractChannel, Btop, Atop, ω, V)
    @tullio g[k,a] := ∂ω_ϕout(act, Btop[k,a], Atop[k,a], ω[k,a], V[k,a])  avx=false
    return g
end

function compute_gcav(act::AbstractChannel, Btop, Atop, ωcav, V)
    @tullio gcav[k,i,a] := ∂ω_ϕout(act, Btop[k,a], Atop[k,a], ωcav[k,i,a], V[k,a])  avx=false
   return gcav
end


channel(ch::AbstractChannel) = ch
channel(arg::Tuple) = (@assert length(arg)==2; channel(arg[1]; arg[2]...))


channel(name::Symbol; prms...) = name == :sign ? ActSign(; name=name, prms...) :
                                 name == :relu  ? ActReLU(; name=name, prms...) : error("no such channel $name")
                                #  name == :id   ? ActId(; name=name, prms...) :
                                #  name == :abs  ? ActAbs(; name=name, prms...) :
                                #  name == :L0   ? RegL0(; name=name, prms...) :
                                #  name == :L1   ? RegL1(; name=name, prms...) :
                                #  name == :Lp   ? RegLp(; name=name, prms...) :
                                #  name == :gauss   ? PriorGauss(; name=name, prms...) :
                                #  name == :radam   ? PriorRadam(; name=name, prms...) :
                                #  name == :berngauss   ? PriorBernGauss(; name=name, prms...) :
                                #  name == :bernradam   ? PriorBernRadam(; name=name, prms...) : error("no such channel $name")

Base.Broadcast.broadcastable(r::AbstractChannel) = Ref(r) # opt-out of broadcast
#####################################