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

    allpu::VecVec # p(σ=up) from fact ↑ to y
    allpd::VecVec # p(σ=up) from y  ↓ to fact

    top_allpd::VecVec
    bottom_allpu::VecVec

    expf::CVec
    expinv0::CVec
    expinv2p::CVec
    expinv2m::CVec
    expinv2P::CVec
    expinv2M::CVec

    top_layer::AbstractLayer
    bottom_layer::AbstractLayer

    allhext::VecVec
    weight_mask::Vector{Vector{Int}}
    isfrozen::Bool
end


function BPExactLayer(K::Int, N::Int, M::Int; density=1, isfrozen=false)
    # for variables W
    allm = [zeros(N) for i=1:K]
    allh = [zeros(N) for i=1:K]
    allhext = [zeros(N) for i=1:K]
    allux = [zeros(N) for i=1:K]


    allmcav = [[zeros(N) for i=1:M] for i=1:K]
    allmycav = [[zeros(N) for i=1:K] for i=1:M]
    allmhcavtoy = [[zeros(K) for i=1:N] for i=1:M]
    allmhcavtow = [[zeros(M) for i=1:N] for i=1:K]
    # for variables Y
    allmy = [zeros(N) for a=1:M]
    allhy = [zeros(N) for a=1:M]

    # for Facts
    allmh = [zeros(M) for k=1:K]

    allpu = [zeros(M) for k=1:K]
    allpd = [zeros(M) for k=1:N]


    expf =fexpf(N)
    expinv0 = fexpinv0(N)
    expinv2p = fexpinv2p(N)
    expinv2m = fexpinv2m(N)
    expinv2P = fexpinv2P(N)
    expinv2M = fexpinv2M(N)

    weight_mask = [[rand() < density ? 1 : 0 for i=1:N] for i=1:K]

    return BPExactLayer(-1, K, N, M, allm, allmy, allmh
        , allmcav, allmycav, allmhcavtoy,allmhcavtow
        , allh, allux, allhy, allpu,allpd
        , VecVec(), VecVec()
        , fexpf(N), fexpinv0(N), fexpinv2p(N), fexpinv2m(N), fexpinv2P(N), fexpinv2M(N)
        , DummyLayer(), DummyLayer(),
        allhext, weight_mask, isfrozen)
end


function updateFact!(layer::BPExactLayer, k::Int, a::Int, reinfpar)
    @extract layer: K N M allm allmy allmh allpu allpd
    @extract layer: bottom_allpu top_allpd
    @extract layer: expf expinv0 expinv2M expinv2P expinv2m expinv2p
    @extract layer: allmcav allmycav allmhcavtow allmhcavtoy
    #TODO add reinforcement/dumping

    mh = allmh[k];
    pdtop = top_allpd[k];
    mycav = allmycav[a][k]
    mcav = allmcav[k][a]
    mhw = allmhcavtow[k]
    mhy = allmhcavtoy[a]
    mask = layer.weight_mask[k]

    X = ones(Complex{Float64}, N+1)
    for p=1:N+1
        for i=1:N
            pup = (1+mcav[i]*mycav[i])/2 * mask[i]
            X[p] *= (1-pup) + pup*expf[p]
        end
    end

    vH = tanh(pdtop[a])
    if !isfrozen(layer)
        s2P = Complex{Float64}(0.)
        s2M = Complex{Float64}(0.)
        for p=1:N+1
            s2P += expinv2P[p] * X[p]
            s2M += expinv2M[p] * X[p]
        end
        s2PP = abs(real(s2P)) / (abs(real(s2P)) + abs(real(s2M)))
        s2MM = abs(real(s2M)) / (abs(real(s2P)) + abs(real(s2M)))
        allpu[k][a] = myatanh(s2PP, s2MM)
        mh[a] = ((1+vH)*s2PP - (1-vH)*s2MM) / ((1+vH)*s2PP + (1-vH)*s2MM)
    end

    for i = 1:N
        mask[i] == 1 || continue
        pup = (1+mcav[i]*mycav[i])/2
        s0 = Complex{Float64}(0.)
        s2p = Complex{Float64}(0.)
        s2m = Complex{Float64}(0.)
        for p=1:N+1
            xp = X[p] / (1-pup + pup*expf[p])
            s0 += expinv0[p] * xp
            s2p += expinv2p[p] * xp
            s2m += expinv2m[p] * xp
        end
        pp = (1+vH)/2; pm = 1-pp
        sr = vH * real(s0 / (pp*(s0 + 2s2p) + pm*(s0 + 2s2m)))
        sr > 1 && (sr=1 - 1e-10) #print("!")
        sr < -1 && (sr=-1 + 1e-10) #print("!")
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
        @assert isfinite(allpd[i][a])
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

    allpu::VecVec # p(σ=up) from fact ↑ to y
    allpd::VecVec # p(σ=up) from y  ↓ to fact

    top_allpd::VecVec
    bottom_allpu::VecVec

    top_layer::AbstractLayer
    bottom_layer::AbstractLayer

    allhext::VecVec
    weight_mask::Vector{Vector{Int}}
    isfrozen::Bool
end

function BPAccurateLayer(K::Int, N::Int, M::Int; density=1, isfrozen=false)
    # for variables W
    allm = [zeros(N) for i=1:K]
    allh = [zeros(N) for i=1:K]
    allhext = [zeros(N) for i=1:K]
    allux = [zeros(N) for i=1:K]

    allmcav = [[zeros(N) for i=1:M] for i=1:K]
    allmycav = [[zeros(N) for i=1:K] for i=1:M]
    allmhcavtoy = [[zeros(K) for i=1:N] for i=1:M]
    allmhcavtow = [[zeros(M) for i=1:N] for i=1:K]
    # for variables Y
    allmy = [zeros(N) for a=1:M]
    allhy = [zeros(N) for a=1:M]

    # for Facts
    allmh = [zeros(M) for k=1:K]

    allpu = [zeros(M) for k=1:K]
    allpd = [zeros(M) for k=1:N]

    weight_mask = [[rand() < density ? 1 : 0 for i=1:N] for i=1:K]

    return BPAccurateLayer(-1, K, N, M, allm, allmy, allmh
        , allmcav, allmycav, allmhcavtoy, allmhcavtow
        , allh, allux, allhy, allpu,allpd
        , VecVec(), VecVec()
        , DummyLayer(), DummyLayer(),
        allhext, weight_mask, isfrozen)
end


function updateFact!(layer::BPAccurateLayer, k::Int, a::Int, reinfpar)
    @extract layer: K N M allm allmy allmh allpu allpd
    @extract layer: bottom_allpu top_allpd
    @extract layer: allmcav allmycav allmhcavtow allmhcavtoy

    mh = allmh[k];
    pd = top_allpd[k];
    my = allmycav[a][k]
    m = allmcav[k][a]
    mhw = allmhcavtow[k]
    mhy = allmhcavtoy[a]
    Mhtot = 0.
    Chtot = 0.
    mask = layer.weight_mask[k]
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

    Chtot == 0 &&  (Chtot = 1e-8); # print("!")

    # println("Mhtot $a= $Mhtot pd=$(pd[a])")
    # @assert isfinite(pd[a]) "$(pd)"
    # if pd[a]*Hp + (1-pd[a])*Hm <= 0.
    #     pd[a] -= 1e-8
    # end
    mh[a] = 1/√Chtot * GH(pd[a], -Mhtot / √Chtot)
    #@show Chtot mh[a]
    @assert isfinite(mh[a])
    if !isbottomlayer(layer)
        for i=1:N
            mask[i] == 1 || continue
            Mcav = Mhtot - my[i]*m[i]
            Ccav = sqrt(Chtot - (1-my[i]^2 * m[i]^2))
            Ccav == 0 &&  (Ccav = 1e-8); # print("!")
            sdσ̄² = √2 * Ccav
            m₊ = (Mcav + my[i]) / sdσ̄²
            m₋ = (Mcav - my[i]) / sdσ̄²
            my₊ = (Mcav + m[i]) / sdσ̄²
            my₋ = (Mcav - m[i]) / sdσ̄²
            h = clamp(pd[a], -30, +30) |> Magnetizations.f2mT
            mhw[i][a] = reinfpar.ψ * mhw[i][a] + (1-reinfpar.ψ) * erfmix(h, m₊, m₋)
            mhy[i][k] = erfmix(h, my₊, my₋)
            @assert isfinite(mhy[i][k]) "isfinite(mhy[i][k]) Ccav=$Ccav"
        end
    else
        for i=1:N
            mask[i] == 1 || continue
            Mcav = Mhtot - my[i]*m[i]
            Ccav = sqrt(Chtot - my[i]^2*(1-m[i]^2))
            Ccav == 0 &&  (Ccav = 1e-8)
            sdσ̄² = √2 * Ccav
            m₊ = (Mcav + my[i]) / sdσ̄²
            m₋ = (Mcav - my[i]) / sdσ̄²
            h = clamp(pd[a], -30, +30) |> Magnetizations.f2mT
            mhw[i][a] = reinfpar.ψ * mhw[i][a] + (1-reinfpar.ψ) * erfmix(h, m₊, m₋)
            @assert isfinite(mhw[i][a]) "m₊ = $(m₊)  m₋ = $(m₋) pd[a]=$(pd[a])"
        end
    end

    allpu[k][a] = atanh2Hm1(-Mhtot / √Chtot)
end


#############################################################
#   BPLayer
##############################################################

mutable struct BPLayer <: AbstractLayer
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

    allpu::VecVec # p(σ=up) from fact ↑ to y
    allpd::VecVec # p(σ=up) from y  ↓ to fact

    top_allpd::VecVec
    bottom_allpu::VecVec

    top_layer::AbstractLayer
    bottom_layer::AbstractLayer

    allhext::VecVec
    weight_mask::Vector{Vector{Int}}
    isfrozen::Bool
end


function BPLayer(K::Int, N::Int, M::Int; density=1., isfrozen=false)
    # for variables W
    allm = [zeros(N) for i=1:K]
    allh = [zeros(N) for i=1:K]
    allhext = [zeros(N) for i=1:K]
    allux = [zeros(N) for i=1:K]

    allmcav = [[zeros(N) for i=1:M] for i=1:K]
    allmycav = [[zeros(N) for i=1:K] for i=1:M]
    allmhcavtoy = [[zeros(K) for i=1:N] for i=1:M]
    allmhcavtow = [[zeros(M) for i=1:N] for i=1:K]
    # for variables Y
    allmy = [zeros(N) for a=1:M]
    allhy = [zeros(N) for a=1:M]

    # for Facts
    allmh = [zeros(M) for k=1:K]

    allpu = [zeros(M) for k=1:K]
    allpd = [zeros(M) for k=1:N]

    weight_mask = [[rand() < density ? 1 : 0 for i=1:N] for i=1:K]

    return BPLayer(-1, K, N, M, allm, allmy, allmh
        , allmcav, allmycav, allmhcavtoy,allmhcavtow
        , allh, allux, allhy, allpu,allpd
        , VecVec(), VecVec()
        , DummyLayer(), DummyLayer(),
        allhext, weight_mask, isfrozen)
end

function updateFact!(layer::BPLayer, k::Int, a::Int, reinfpar)
#function updateFact!(layer::BPLayer, k::Int)
    @extract layer: K N M allm allmy allmh allpu allpd
    @extract layer: bottom_allpu top_allpd
    @extract layer: allmcav allmycav allmhcavtow allmhcavtoy

    mh = allmh[k];
    pd = top_allpd[k];

    my = allmycav[a][k]
    m = allmcav[k][a]
    mhw = allmhcavtow[k]
    mhy = allmhcavtoy[a]
    Mhtot = 0.
    Chtot = 1e-10
    mask = layer.weight_mask[k]
    if !isbottomlayer(layer)
        for i=1:N
            Mhtot += my[i]*m[i] * mask[i]
            Chtot += (1 - my[i]^2*m[i]^2) * mask[i]
        end
    else
        for i=1:N
            Mhtot += my[i]*m[i] * mask[i]
            Chtot += (my[i]^2*(1 - m[i]^2)) * mask[i]
        end
    end

    mh[a] = 1/√Chtot * GH(pd[a], -Mhtot / √Chtot)

    @assert isfinite(mh[a])
    if !isbottomlayer(layer)
        for i=1:N
            mask[i] == 1 || continue
            Mcav = Mhtot - my[i]*m[i]
            Ccav = sqrt(Chtot - (1-my[i]^2 * m[i]^2))
            gh = GH(pd[a],-Mcav / Ccav)
            @assert isfinite(gh)

            mhw[i][a] = (1-reinfpar.ψ) * my[i]/Ccav * gh  + reinfpar.ψ * mhw[i][a]
            mhy[i][k] = (1-reinfpar.ψ) * m[i]/Ccav * gh   + reinfpar.ψ * mhy[i][k]
            @assert isfinite(mhy[i][k]) "isfinite(mhy[i][k]) gh=$gh Ccav=$Ccav"
        end
    else
        for i=1:N
            mask[i] == 1 || continue
            Mcav = Mhtot - my[i]*m[i]
            Ccav = sqrt(Chtot - my[i]^2*(1-m[i]^2))
            gh = GH(pd[a], -Mcav / Ccav)
            @assert isfinite(gh)
            mhw[i][a] = (1-reinfpar.ψ) * my[i]/Ccav * gh  + reinfpar.ψ * mhw[i][a]
        end
    end

    allpu[k][a] = atanh2Hm1(-Mhtot / √Chtot)
end

function updateVarW!(layer::L, k::Int, i::Int, reinfpar) where {L <: Union{BPLayer, BPAccurateLayer, BPExactLayer}}
    @extract layer: K N M allm allmy allmh allpu allpd allh allux allhext
    @extract layer: bottom_allpu top_allpd
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
    if layer.weight_mask[k][i] == 0
        @assert m[i] == 0 "m[i]=$(m[i]) should be 0"
    end
    for a=1:M
        mcav[a][i] = tanh(h[i]-mhw[a])
        #@assert isfinite(h[i])
        #@assert isfinite(mhw[a])
        #@assert isfinite(mcav[a][i])
    end
    Δ = max(Δ, abs(m[i] - oldm))
    return Δ
end

function updateVarY!(layer::L, a::Int, ry::Float64=0.) where {L <: Union{BPLayer, BPAccurateLayer, BPExactLayer}}
    @extract layer: K N M allm allmy allmh allpu allpd allhy
    @extract layer: bottom_allpu top_allpd
    @extract layer: allmcav allmycav allmhcavtow allmhcavtoy

    @assert !isbottomlayer(layer)

    my = allmy[a]
    hy = allhy[a]

    for i=1:N
        mhy = allmhcavtoy[a][i]
        mycav = allmycav[a]
        hy[i] = sum(mhy) + ry* hy[i]
        # @assert isfinite(hy[i]) "isfinite(hy[i]) mhy=$mhy"
        allpd[i][a] = hy[i]
        # (allpd[i][a] < 0.) && (print("!y");allpd[i][a] = 1e-10)
        # (allpd[i][a] > 1.) && (print("!y");allpd[i][a] = 1-1e-10)
        # @assert isfinite(allpd[i][a]) "isfinite(allpd[i][a]) $(MYt[i]) $(my[i] * CYt) $(hy[i])"
        # pinned from below (e.g. from input layer)
        # if pu > 1-1e-10 || pu < 1e-10
        #     hy[i] = pu > 0.5 ? 100 : -100
        #     my[i] = 2pu-1
        #     for k=1:K
        #         mycav[k][i] = 2pu-1
        #     end
        # else
        pu = bottom_allpu[i][a];
        hy[i] += pu
        my[i] = tanh(hy[i])
        @assert isfinite(my[i]) "isfinite(my[i]) pu=$pu mhy=$mhy $(typeof(layer)))"
        for k=1:K
            mycav[k][i] = tanh(hy[i]-mhy[k])
        end
    end
end

function initYBottom!(layer::L, a::Int, ry::Float64=0.) where {L <: Union{BPLayer, BPAccurateLayer, BPExactLayer}}
    @extract layer: K N M allm allmy allmh allpu allpd allhy
    @extract layer: bottom_allpu top_allpd
    @extract layer: allmcav allmycav allmhcavtow allmhcavtoy

    @assert isbottomlayer(layer)

    my = allmy[a]
    x = layer.bottom_layer.x
    for i=1:N
        my[i] = x[i, a]
        mycav = allmycav[a]
        for k=1:K
            mycav[k][i] = x[i, a]
        end
    end
end


function update!(layer::L, reinfpar) where {L <: Union{BPLayer, BPAccurateLayer, BPExactLayer}}
    @extract layer: K N M allm allmy allmh allpu allpd allhy
    @extract layer: bottom_allpu top_allpd
    @extract layer: allmcav allmycav allmhcavtow allmhcavtoy

    # @show allm allmy allmh allpu allpd allhy top_allpd
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
            updateVarY!(layer, a, reinfpar.ry)
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
    #         updateVarY!(layer, a, reinfpar.ry)
    #     end
    # end


    return Δ
end


function initrand!(layer::L) where {L <: Union{BPLayer, BPAccurateLayer, BPExactLayer}}
    @extract layer K N M allm allmy allmh allpu allpd  top_allpd
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy
    ϵ = 1e-1
    mask = layer.weight_mask

    for (k, m) in enumerate(allm)
        m .= ϵ*(2*rand(N) .- 1) .* mask[k]
    end
    for my in allmy
        my .= ϵ*(2*rand(N) .- 1)
    end
    for mh in allmh
        mh .= ϵ*(2*rand(M) .- 1)
    end
    for pu in allpu
        pu .= rand(M)
    end
    for pd in allpd
        pd .= rand(M)
    end

    # if!isbottomlayer
    for k=1:K,a=1:M,i=1:N
        allmcav[k][a][i] = allm[k][i] * mask[k][i]
        allmycav[a][k][i] = allmy[a][i]
        allmhcavtow[k][i][a] = allmh[k][a]*allmy[a][i] * mask[k][i]
        allmhcavtoy[a][i][k] = allmh[k][a]*allm[k][i] * mask[k][i]
    end

end

function fixW!(layer::L, w=1.) where {L <: Union{BPLayer, BPAccurateLayer, BPExactLayer}}
    @extract layer K N M allm allmy allmh allpu allpd top_allpd
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

    for k=1:K,i=1:N
        allm[k][i] = w
    end
    for k=1:K, a=1:M, i=1:N
        allmcav[k][a][i] = allm[k][i]
    end
end

function fixY!(layer::L, x::Matrix) where {L <: Union{BPLayer, BPAccurateLayer, BPExactLayer}}
    @extract layer: K N M allm allmy allmh allpu allpd top_allpd
    @extract layer: allmcav allmycav allmhcavtow allmhcavtoy

    for a=1:M,i=1:N
        allmy[a][i] = x[i,a]
    end
    for a=1:M, k=1:K, i=1:N
        allmycav[a][k][i] = allmy[a][i]
    end
end

function getW(layer::L) where L <: Union{BPLayer, BPAccurateLayer, BPExactLayer}
    @extract layer: weight_mask allm K
    return vcat([(sign.(allm[k] .+ 1e-10) .* weight_mask[k])' for k in 1:K]...)
end

function forward(layer::L, x) where L <: Union{BPLayer, BPAccurateLayer, BPExactLayer}
    @extract layer: N K
    @assert size(x, 1) == N
    W = getW(layer)
    return sign.(W*x .+ 1e-10)
end

