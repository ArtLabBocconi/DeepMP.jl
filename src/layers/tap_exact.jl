
###########################
#       TAP EXACT LAYER
#######################################
mutable struct TapExactLayer <: AbstractLayer
    l::Int
    K::Int
    N::Int
    M::Int

    allm::VecVec
    allmy::VecVec
    allmh::VecVec

    allh::VecVec # for W reinforcement
    allhext::VecVec # for W reinforcement
    allhy::VecVec # for Y reinforcement

    Bup  # field from fact  ↑ to y
    B # field from y ↓ to fact
    Mtot::VecVec
    Ctot::Vec
    MYtot::VecVec
    CYtot::Vec

    expf::CVec
    expinv0::CVec
    expinv2p::CVec
    expinv2m::CVec
    expinv2P::CVec
    expinv2M::CVec

    top_layer::AbstractLayer
    bottom_layer::AbstractLayer

    weight_mask
    isfrozen::Bool
end

function TapExactLayer(K::Int, N::Int, M::Int; density=1, isfrozen=false)
    # for variables W
    allm = [zeros(F, N) for i=1:K]
    allh = [zeros(F, N) for i=1:K]
    Mtot = [zeros(F, N) for i=1:K]
    Ctot = zeros(F, K)
    allhext = [zeros(F, N) for i=1:K]
    
    # for variables Y
    allmy = [zeros(F, N) for a=1:M]
    allhy = [zeros(F, N) for a=1:M]
    MYtot = [zeros(F, N) for a=1:M]
    CYtot = zeros(F, M)

    # for Facts
    allmh = [zeros(F, M) for k=1:K]

    Bup = zeros(F, K, M)
    B = zeros(F, N, M)


    expf =fexpf(N)
    expinv0 = fexpinv0(N)
    expinv2p = fexpinv2p(N)
    expinv2m = fexpinv2m(N)
    expinv2P = fexpinv2P(N)
    expinv2M = fexpinv2M(N)

    weight_mask = rand(F, K, N) .< density

    return TapExactLayer(-1, K, N, M, allm, allmy, allmh 
        , allh, allhext, allhy, Bup, B
        , Mtot, Ctot, MYtot, CYtot
        , fexpf(N), fexpinv0(N), fexpinv2p(N), fexpinv2m(N), fexpinv2P(N), fexpinv2M(N)
        , DummyLayer(), DummyLayer(),
        weight_mask, isfrozen)
end


## Utility fourier tables for the exact theta node
fexpf(N) = Complex{F}[exp(2π*im*p/(N+1)) for p=0:N]
fexpinv0(N) = Complex{F}[exp(-2π*im*p*(N-1)/2/(N+1)) for p=0:N]
fexpinv2p(N) = Complex{F}[(
        a =exp(-2π*im*p*(N-1)/2/(N+1));
        b = exp(-2π*im*p/(N+1));
        p==0 ? (N+1)/2 : a*b/(1-b)*(1-b^((N+1)/2)))
        for p=0:N]
fexpinv2m(N) = Complex{F}[(
        a =exp(-2π*im*p*(N-1)/2/(N+1));
        b = exp(2π*im*p/(N+1));
        p==0 ? (N+1)/2 : a*b/(1-b)*(1-b^((N+1)/2)))
        for p=0:N]
fexpinv2P(N) = Complex{F}[(
        a =exp(-2π*im*p/(N+1)*(N+1)/2);
        b = exp(-2π*im*p/(N+1));
        p==0 ? (N+1)/2 : a/(1-b)*(1-b^((N+1)/2)))
        for p=0:N]
fexpinv2M(N) = Complex{F}[(
        a =exp(-2π*im*p/(N+1)*(N-1)/2);
        b = exp(2π*im*p/(N+1));
        p==0 ? (N+1)/2 : a/(1-b)*(1-b^((N+1)/2)))
        for p=0:N]



function updateFact!(layer::TapExactLayer, k::Int, reinfpar)
    @extract layer: K N M allm allmy allmh B Bup
    @extract layer: CYtot MYtot Mtot Ctot
    @extract layer: bottom_layer top_layer
    @extract layer: expf expinv0 expinv2M expinv2P expinv2m expinv2p
    m = allm[k]; mh = allmh[k]
    Mt = Mtot[k]; Ct = Ctot
    pdtop = top_layer.B[k,:]
    CYt = CYtot
    for a=1:M
        my = allmy[a]
        MYt = MYtot[a];
        X = ones(Complex{F}, N+1)
    
        for p=1:N+1
            for i=1:N
                # magY = my[i]-mh[a]*m[i]*(1-my[i]^2)
                # magW = m[i]-mh[a]*my[i]*(1-m[i]^2)
                # pup = (1+magY*magW)/2
                pup = (1+my[i]*m[i])/2
                X[p] *= (1-pup) + pup*expf[p]
            end
        end
        
        vH = tanh(pdtop[a])
        s2P = Complex{F}(0.)
        s2M = Complex{F}(0.)
        for p=1:N+1
            s2P += expinv2P[p] * X[p]
            s2M += expinv2M[p] * X[p]
        end
        s2PP = abs(real(s2P)) / (abs(real(s2P)) + abs(real(s2M)))
        s2MM = abs(real(s2M)) / (abs(real(s2P)) + abs(real(s2M)))
        Bup[k,a] = myatanh(s2PP, s2MM)
        mh[a] = real((1+vH)*s2P - (1-vH)*s2M) / real((1+vH)*s2P + (1-vH)*s2M)

        for i = 1:N
            # magY = istoplayer ? 2pubot[i][a]-1 : my[i]-mh[a]*m[i]*(1-my[i]^2)
            # magW = m[i]-mh[a]*my[i]*(1-m[i]^2)
            magY = my[i]
            magW = m[i]
            pup = (1+magY*magW)/2

            s0 = Complex{F}(0.)
            s2p = Complex{F}(0.)
            s2m = Complex{F}(0.)
            for p=1:N+1
                xp = X[p] / (1-pup + pup*expf[p])
                s0 += expinv0[p] * xp
                s2p += expinv2p[p] * xp
                s2m += expinv2m[p] * xp
            end
            pp = (1+vH)/2; pm = 1-pp
            sr = vH * real(s0 / (pp*(s0 + 2s2p) + pm*(s0 + 2s2m)))
            @assert isfinite(sr)
            sr > 1 && (sr=1. - 1f-12) #print("!")
            sr < -1 && (sr=-1. + 1f-12) #print("!")
            # if isfrozen(layer)
            #     B[i,a] = atanh(m[i]*sr)
            # else
            MYt[i] +=  myatanh(m[i] * sr)
            Mt[i] +=  myatanh(my[i] * sr)
            # end
            # @assert isfinite(my[i])
            # @assert isfinite(B[i,a])
            @assert isfinite(MYt[i])
        end
    end
end

function updateVarW!(layer::L, k::Int, r=0) where {L <: Union{TapExactLayer}}
    @extract layer: K N M allm allmy allmh B Bup l
    @extract layer: CYtot MYtot Mtot Ctot allh allhext
    Δ = 0.
    m=allm[k];
    Mt=Mtot[k]; Ct = Ctot;
    h=allh[k]
    hext = allhext[k]
    for i=1:N
        if layer.weight_mask[k,i] == 0
            @assert m[i] == 0 "m[i]=$(m[i]) shiuld be 0"
        end
        h[i] = Mt[i] + m[i] * Ct[k] + r*h[i] + hext[i]
        oldm = m[i]
        m[i] = tanh(h[i])
        Δ = max(Δ, abs(m[i] - oldm))
    end
    return Δ
end

function updateVarY!(layer::L, a::Int) where {L <: Union{TapExactLayer}}
    @extract layer: K N M allm allmy allmh B Bup
    @extract layer: allhy CYtot MYtot Mtot Ctot
    @extract layer: bottom_layer
    
    @assert !isbottomlayer(layer)

    MYt=MYtot[a]; CYt = CYtot[a]; my=allmy[a]; hy=allhy[a]
    @assert isfinite(CYt) "CYt=$CYt"
    for i=1:N
        @assert isfinite(MYt[i]) "MYt[i]=$(MYt[i]) "
        @assert isfinite(my[i]) "my[i]=$(my[i]) "
        #TODO inutile calcolarli per il primo layer
        @assert isfinite(hy[i])
        hy[i] = MYt[i] + my[i] * CYt
        @assert isfinite(hy[i]) "MYt[i]=$(MYt[i]) my[i]=$(my[i]) CYt=$CYt hy[i]=$(hy[i])"
        B[i,a] = hy[i]
        # @assert isfinite(B[i,a]) "isfinite(B[i,a]) $(MYt[i]) $(my[i] * CYt) $(hy[i])"
        # pinned from below (e.g. from input layer)
        # if pu > 1-1f-10 || pu < 1f-10 # NOTE:1-e15 dà risultati peggiori
        #     hy[i] = pu > 0.5 ? 100 : -100
        #     my[i] = 2pu-1
        #
        # else
        pu = bottom_layer.Bup[i,a]
        hy[i] += pu
        @assert isfinite(hy[i]) "pu=$pu layer.l=$(layer.l)"
        my[i] = tanh(hy[i])
        @assert isfinite(my[i]) "isfinite(my[i]) pu=$pu hy[i]=$(hy[i])"
        # end
    end
end

function update!(layer::L, reinfpar; mode=:both) where {L <: Union{TapExactLayer}}
    @extract layer K N M allm allmy allmh B Bup CYtot MYtot Mtot Ctot


    #### Reset Total Fields
    CYtot .= 0
    for a=1:M
        MYtot[a] .= 0
    end
    for k=1:K
        Mtot[k] .= 0
        Ctot[k] = 0
    end
    ############

    for k=1:K
        updateFact!(layer, k, reinfpar)
    end
    Δ = 0.
    if !isfrozen(layer)
        for k=1:K
            δ = updateVarW!(layer, k, reinfpar.r)
            Δ = max(δ, Δ)
        end
    end

    # bypass Y if top_layer
    if !isbottomlayer(layer)
        for a=1:M
            updateVarY!(layer, a)
        end
    end
    return Δ
end

function initrand!(layer::L) where {L <: Union{TapExactLayer}}
    @extract layer K N M allm allmy allmh B Bup
    ϵ = 1f-1
    mask = layer.weight_mask

    for (k, m) in enumerate(allm)
        m .= (2*rand(F, N) .- 1) .* ϵ .* mask[k,:]
    end
    for my in allmy
        my .= (2*rand(F, N) .- 1) .* ϵ
    end
    for mh in allmh
        mh .= (2*rand(F, M) .- 1) .* ϵ
    end
end

function fixW!(layer::L, w=1.) where {L <: Union{TapExactLayer}}
    @extract layer K N M allm allmy allmh B Bup

    for k=1:K, i=1:N
        allm[k][i] = w
    end
end

function fixY!(layer::L, x::AbstractMatrix) where {L <: Union{TapExactLayer}}
    @extract layer K N M allm allmy allmh B Bup

    for a=1:M, i=1:N
        allmy[a][i] = x[i,a]
    end
end

function getW(layer::L) where L <: Union{TapExactLayer}
    @extract layer: weight_mask allm K
    return vcat([(sign.(allm[k] .+ 1f-10) .* weight_mask[k,:])' for k in 1:K]...)
end

function forward(layer::L, x) where L <: Union{TapExactLayer}
    @extract layer: N K
    @assert size(x, 1) == N
    W = getW(layer)
    @assert size(W) == (K, N)
    return sign.(W*x .+ 1f-10)
end
