##### CHANNELS  ##################
abstract type AbstractChannel end

∂ω_ϕout(ch::AbstractChannel, y, ω, V) = deriv(ω->ϕout(ch, y, ω, V), ω)
∂²ω_ϕout(ch::AbstractChannel, y, ω, V, chout::AbstractChannel) = deriv(ω->∂ω_ϕout(ch, y, ω, V), ω)
∂B_ϕin(ch::AbstractChannel, B, A, chin::AbstractChannel) = deriv(B->ϕin(ch, B, A), B)
∂²B_ϕin(ch::AbstractChannel, B, A, chin::AbstractChannel) = deriv(B->∂B_ϕin(ch, B, A), B)


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
