#TODO Layer not working


mutable struct BPExactLayer <: AbstractLayer
    l::Int
    K::Int
    N::Int
    M::Int

    allm::VecVec
    allmy::VecVec
    allmh::VecVec

    allmcav::VecVecVec
    allmycav::VecVecVec
    allmhcavtoy::VecVecVec
    allmhcavtow::VecVecVec

    allh::VecVec # for W reinforcement
    allux::VecVec # for focusing
    allhy::VecVec # for Y reinforcement

    Bup
    B

    expf::CVec
    expinv0::CVec
    expinv2p::CVec
    expinv2m::CVec
    expinv2P::CVec
    expinv2M::CVec

    top_layer::AbstractLayer
    bottom_layer::AbstractLayer

    allhext::VecVec
    weight_mask
    isfrozen::Bool
end


function BPExactLayer(K::Int, N::Int, M::Int; density=1, isfrozen=false)
    # for variables W
    allm = [zeros(F, N) for i=1:K]
    allh = [zeros(F, N) for i=1:K]
    allhext = [zeros(F, N) for i=1:K]
    allux = [zeros(F, N) for i=1:K]


    allmcav = [[zeros(F, N) for i=1:M] for i=1:K]
    allmycav = [[zeros(F, N) for i=1:K] for i=1:M]
    allmhcavtoy = [[zeros(F, K) for i=1:N] for i=1:M]
    allmhcavtow = [[zeros(F, M) for i=1:N] for i=1:K]
    # for variables Y
    allmy = [zeros(F, N) for a=1:M]
    allhy = [zeros(F, N) for a=1:M]

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

    return BPExactLayer(-1, K, N, M, allm, allmy, allmh
        , allmcav, allmycav, allmhcavtoy,allmhcavtow
        , allh, allux, allhy, Bup, B
        , fexpf(N), fexpinv0(N), fexpinv2p(N), fexpinv2m(N), fexpinv2P(N), fexpinv2M(N)
        , DummyLayer(), DummyLayer(),
        allhext, weight_mask, isfrozen)
end


function updateFact!(layer::BPExactLayer, k::Int, a::Int, reinfpar)
    @extract layer: K N M allm allmy allmh B Bup
    @extract layer: expf expinv0 expinv2M expinv2P expinv2m expinv2p
    @extract layer: allmcav allmycav allmhcavtow allmhcavtoy
    @extract layer: bottom_layer top_layer
    #TODO add reinforcement/dumping

    mh = allmh[k]
    vH = tanh(top_layer.B[k,a])
    mycav = allmycav[a][k]
    mcav = allmcav[k][a]
    mhw = allmhcavtow[k]
    mhy = allmhcavtoy[a]
    mask = layer.weight_mask[k,:]

    X = ones(Complex{F}, N+1)
    for p=1:N+1
        for i=1:N
            pup = (1+mcav[i]*mycav[i])/2 * mask[i]
            X[p] *= (1-pup) + pup*expf[p]
        end
    end

    if !isfrozen(layer)
        s2P = Complex{F}(0.)
        s2M = Complex{F}(0.)
        for p=1:N+1
            s2P += expinv2P[p] * X[p]
            s2M += expinv2M[p] * X[p]
        end
        s2PP = abs(real(s2P)) / (abs(real(s2P)) + abs(real(s2M)))
        s2MM = abs(real(s2M)) / (abs(real(s2P)) + abs(real(s2M)))
        Bup[k,a] = myatanh(s2PP, s2MM)
        mh[a] = ((1+vH)*s2PP - (1-vH)*s2MM) / ((1+vH)*s2PP + (1-vH)*s2MM)
    end

    for i = 1:N
        mask[i] == 1 || continue
        pup = (1+mcav[i]*mycav[i])/2
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
        sr > 1 && (sr=1 - 1f-10) #print("!")
        sr < -1 && (sr=-1 + 1f-10) #print("!")
        if !isfrozen(layer)
            mhw[i][a] =  myatanh(mycav[i] * sr)
            !isfinite(mhw[i][a]) && (mhw[i][a] = sign(mhw[i][a])*20) #print("!")
            @assert isfinite(mhw[i][a]) "mhw[i][a]=$(mhw[i][a]) $(mycav[i]) $sr"
        end
        if !isbottomlayer(layer)
            mhy[i][k] =  myatanh(mcav[i] * sr)
            !isfinite(mhy[i][k]) && (mhy[i][k] = sign(mhy[i][k])*20); #print("!")
            @assert isfinite(mhy[i][k]) "mhy[i][k]=$(mhy[i][k]) $(mcav) $sr"
        end
        @assert isfinite(mycav[i])
        @assert isfinite(B[i,a])
        @assert isfinite(sr)
    end
end


#############################################################
#   BPAccurateLayer
##############################################################

mutable struct BPAccurateLayer <: AbstractLayer
    l::Int
    K::Int
    N::Int
    M::Int

    allm::VecVec
    allmy::VecVec
    allmh::VecVec

    allmcav::VecVecVec
    allmycav::VecVecVec
    allmhcavtoy::VecVecVec
    allmhcavtow::VecVecVec

    allh::VecVec # for W reinforcement
    allux::VecVec # for focusing
    allhy::VecVec # for Y reinforcement

    Bup  # field from fact  ↑ to y
    B # field from y ↓ to fact
    
    top_layer::AbstractLayer
    bottom_layer::AbstractLayer

    allhext::VecVec
    weight_mask
    isfrozen::Bool
end

function BPAccurateLayer(K::Int, N::Int, M::Int; density=1, isfrozen=false)
    # for variables W
    allm = [zeros(F, N) for i=1:K]
    allh = [zeros(F, N) for i=1:K]
    allhext = [zeros(F, N) for i=1:K]
    allux = [zeros(F, N) for i=1:K]

    allmcav = [[zeros(F, N) for i=1:M] for i=1:K]
    allmycav = [[zeros(F, N) for i=1:K] for i=1:M]
    allmhcavtoy = [[zeros(F, K) for i=1:N] for i=1:M]
    allmhcavtow = [[zeros(F, M) for i=1:N] for i=1:K]
    # for variables Y
    allmy = [zeros(F, N) for a=1:M]
    allhy = [zeros(F, N) for a=1:M]

    # for Facts
    allmh = [zeros(F, M) for k=1:K]

    Bup = zeros(F, K, M)
    B = zeros(F, N, M)

    weight_mask = rand(F, K, N) .< density

    return BPAccurateLayer(-1, K, N, M, allm, allmy, allmh
        , allmcav, allmycav, allmhcavtoy, allmhcavtow
        , allh, allux, allhy, Bup, B
        , DummyLayer(), DummyLayer(),
        allhext, weight_mask, isfrozen)
end


function updateFact!(layer::BPAccurateLayer, k::Int, a::Int, reinfpar)
    @extract layer: K N M allm allmy allmh B Bup
    @extract layer: bottom_layer top_layer
    @extract layer: allmcav allmycav allmhcavtow allmhcavtoy

    mh = allmh[k]
    pd = top_layer.B[k,a]
    my = allmycav[a][k]
    m = allmcav[k][a]
    mhw = allmhcavtow[k]
    mhy = allmhcavtoy[a]
    Mhtot = 0.
    Chtot = 0.
    mask = layer.weight_mask[k,:]
    if !isbottomlayer(layer)
        for i=1:N
            Mhtot += my[i]*m[i] * mask[i]
            Chtot += (1 - my[i]^2*m[i]^2) * mask[i]
        end
        #println("notbottom, Chtot = $Chtot")
    else
        #println("isbot(0), Chtot = $Chtot, summ=$(sum(m)), summy=$(sum(my))")
        for i=1:N
            Mhtot += my[i]*m[i] * mask[i]
            Chtot += (my[i]^2*(1 - m[i]^2)) * mask[i]
        end
        #println("isbot, Chtot = $Chtot")
    end

    Chtot == 0 &&  (Chtot = 1f-8); # print("!")

    mh[a] = 1/√Chtot * GH2(pd, -Mhtot / √Chtot)
    @assert isfinite(mh[a])
    if !isbottomlayer(layer)
        for i=1:N
            mask[i] == 1 || continue
            Mcav = Mhtot - my[i]*m[i]
            Ccav = sqrt(Chtot - (1-my[i]^2 * m[i]^2))
            Ccav == 0 &&  (Ccav = 1f-8); # print("!")
            sdσ̄² = √2 * Ccav
            m₊ = (Mcav + my[i]) / sdσ̄²
            m₋ = (Mcav - my[i]) / sdσ̄²
            my₊ = (Mcav + m[i]) / sdσ̄²
            my₋ = (Mcav - m[i]) / sdσ̄²
            h = clamp(pd, -30, +30) |> Magnetizations.f2mT
            mhw[i][a] = reinfpar.ψ * mhw[i][a] + (1-reinfpar.ψ) * erfmix(h, m₊, m₋)
            mhy[i][k] = erfmix(h, my₊, my₋)
            @assert isfinite(mhy[i][k]) "isfinite(mhy[i][k]) Ccav=$Ccav"
        end
    else
        for i=1:N
            mask[i] == 1 || continue
            Mcav = Mhtot - my[i]*m[i]
            Ccav = sqrt(Chtot - my[i]^2*(1-m[i]^2))
            Ccav == 0 &&  (Ccav = 1f-8)
            sdσ̄² = √2 * Ccav
            m₊ = (Mcav + my[i]) / sdσ̄²
            m₋ = (Mcav - my[i]) / sdσ̄²
            h = clamp(pd, -30, +30) |> Magnetizations.f2mT
            mhw[i][a] = reinfpar.ψ * mhw[i][a] + (1-reinfpar.ψ) * erfmix(h, m₊, m₋)
            @assert isfinite(mhw[i][a]) "m₊ = $(m₊)  m₋ = $(m₋) pd=$(pd)"
        end
    end

    Bup[k,a] = atanh2Hm1(-Mhtot / √Chtot)
end

function updateVarW!(layer::L, k::Int, i::Int, reinfpar) where {L <: Union{BPAccurateLayer, BPExactLayer}}
    @extract layer: K N M allm allmy allmh B Bup allh allux allhext
    @extract layer: bottom_layer top_layer
    @extract layer: allmcav allmycav allmhcavtow allmhcavtoy

    λ = reinfpar.ψ
    m = allm[k]
    h = allh[k]
    hext = allhext[k]
    ux = allux[k]
    Δ = 0.
    #@assert isfinite(h[i])
    mhw = allmhcavtow[k][i]
    mcav = allmcav[k]
    #@assert isfinite(sum(mhw))

    if reinfpar.y <= 0.0
        h[i] = sum(mhw) + reinfpar.r*h[i] + hext[i]
    else
        pol = tanh(reinfpar.r)
        ρ = reinfpar.y - 1.0
        # ux[i] = λ * ux[i] + (1.0 - λ) * tanh(ρ * atanh( tanh(h[i] - ux[i]) * pol )) * pol # WRONG VERSION
        ux[i] = λ * ux[i] + (1.0 - λ) * atanh(tanh(ρ * atanh( tanh(h[i] - ux[i]) * pol )) * pol)
        h[i] = sum(mhw) + hext[i] + ux[i]
    end
    oldm = m[i]
    m[i] = tanh(h[i])
    if layer.weight_mask[k,i] == 0
        @assert m[i] == 0 "m[i]=$(m[i]) should be 0"
    end
    for a=1:M
        mcav[a][i] = tanh(h[i] - mhw[a])
        #@assert isfinite(h[i])
        #@assert isfinite(mhw[a])
        #@assert isfinite(mcav[a][i])
    end
    Δ = max(Δ, abs(m[i] - oldm))
    return Δ
end

function updateVarY!(layer::L, a::Int) where {L <: Union{BPAccurateLayer, BPExactLayer}}
    @extract layer: K N M allm allmy allmh B Bup allhy
    @extract layer: bottom_layer top_layer
    @extract layer: allmcav allmycav allmhcavtow allmhcavtoy

    @assert !isbottomlayer(layer)

    my = allmy[a]
    hy = allhy[a]

    for i=1:N
        mhy = allmhcavtoy[a][i]
        mycav = allmycav[a]
        hy[i] = sum(mhy)
        # @assert isfinite(hy[i]) "isfinite(hy[i]) mhy=$mhy"
        B[i,a] = hy[i]
        pu = bottom_layer.Bup[i,a]
        hy[i] += pu
        my[i] = tanh(hy[i])
        @assert isfinite(my[i]) "isfinite(my[i]) pu=$pu mhy=$mhy $(typeof(layer)))"
        for k=1:K
            mycav[k][i] = tanh(hy[i]-mhy[k])
        end
    end
end

function update!(layer::L, reinfpar; mode=:both) where {L <: Union{BPAccurateLayer, BPExactLayer}}
    @extract layer: K N M allm allmy allmh B Bup allhy
    @extract layer: bottom_layer top_layer
    @extract layer: allmcav allmycav allmhcavtow allmhcavtoy

    # @show allm allmy allmh B Bup allhy 
    # println("m=$(allm[1])")
    # println("mcav=$(allmcav[1][1])")

    # println("mhcavw=$(allmhcavtow[1][1])")
    Δ = 0.
    for u in randperm(M + N*K)
        if u <= M
            a = u
            for k=1:K
                updateFact!(layer, k, a, reinfpar)
            end
        else
            k = (u-M-1) ÷ N + 1
            i = (u-M-1) % N + 1

            if !isfrozen(layer)
                # println("Updating W")
                δ = updateVarW!(layer, k, i, reinfpar)
                Δ = max(δ, Δ)
            end
        end
    end
    if !isbottomlayer(layer)
        for a=1:M
            updateVarY!(layer, a)
        end
    end

    # Δ = 0.
    # for k in 1:K, a in 1:M
    #     updateFact!(layer, k, a, reinfpar)
    # end

    # for k in 1:K, i in 1:N
    #     if !isfrozen(layer)
    #         # println("Updating W")
    #         δ = updateVarW!(layer, k, i, reinfpar.r)
    #         Δ = max(δ, Δ)
    #     end
    # end
    # if !isbottomlayer(layer)
    #     for a=1:M
    #         updateVarY!(layer, a)
    #     end
    # end


    return Δ
end


function initrand!(layer::L) where {L <: Union{BPAccurateLayer, BPExactLayer}}
    @extract layer K N M allm allmy allmh B Bup 
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy
    ϵ = 1f-1
    mask = layer.weight_mask

    for (k, m) in enumerate(allm)
        m .= ϵ*(2*rand(F, N) .- 1) .* mask[k,:]
    end
    for my in allmy
        my .= ϵ*(2*rand(F, N) .- 1)
    end
    for mh in allmh
        mh .= ϵ*(2*rand(F, M) .- 1)
    end

    # if!isbottomlayer
    for k=1:K,a=1:M,i=1:N
        allmcav[k][a][i] = allm[k][i] * mask[k,i]
        allmycav[a][k][i] = allmy[a][i]
        allmhcavtow[k][i][a] = allmh[k][a] * allmy[a][i] * mask[k,i]
        allmhcavtoy[a][i][k] = allmh[k][a] * allm[k][i] * mask[k,i]
    end

end

function fixW!(layer::L, w=1.) where {L <: Union{BPAccurateLayer, BPExactLayer}}
    @extract layer K N M allm allmy allmh B Bup
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

    for k=1:K,i=1:N
        allm[k][i] = w
    end
    for k=1:K, a=1:M, i=1:N
        allmcav[k][a][i] = allm[k][i]
    end
end

function fixY!(layer::L, x::AbstractMatrix) where {L <: Union{BPAccurateLayer, BPExactLayer}}
    @extract layer: K N M allm allmy allmh B Bup
    @extract layer: allmcav allmycav allmhcavtow allmhcavtoy

    for a=1:M,i=1:N
        allmy[a][i] = x[i,a]
    end
    for a=1:M, k=1:K, i=1:N
        allmycav[a][k][i] = allmy[a][i]
    end
end

function getW(layer::L) where L <: Union{BPAccurateLayer, BPExactLayer}
    @extract layer: weight_mask allm K
    return vcat([(sign.(allm[k] .+ 1f-10) .* weight_mask[k,:])' for k in 1:K]...)
end

function forward(layer::L, x) where L <: Union{BPAccurateLayer, BPExactLayer}
    @extract layer: N K
    @assert size(x, 1) == N
    W = getW(layer)
    return sign.(W*x .+ 1f-10)
end
