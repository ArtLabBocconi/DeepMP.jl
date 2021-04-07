# NOTE le m in realtà sono tutte dei campi
mutable struct MaxSumLayer <: AbstractLayer
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
    allhy::VecVec # for Y reinforcement

    Bup # field from fact ↑ to y
    B # field from y ↓ to fact

    top_layer::AbstractLayer
    bottom_layer::AbstractLayer

    βms::F

    isfrozen::Bool
end

function MaxSumLayer(K::Int, N::Int, M::Int; βms=1., isfrozen=false)
    # for variables W
    allm = [zeros(F, N) for i=1:K]
    allh = [zeros(F, N) for i=1:K]


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

    return MaxSumLayer(-1, K, N, M, allm, allmy, allmh
        , allmcav, allmycav, allmhcavtoy,allmhcavtow
        , allh, allhy, Bup, B
        , DummyLayer(), DummyLayer()
        , βms, isfrozen)
end


function updateVarW!(layer::MaxSumLayer, k::Int, r=1.)
    @extract layer: K N M allm allmy allmh B Bup allhy allh
    @extract layer: bottom_layer top_layer
    @extract layer: allmcav allmycav allmhcavtow allmhcavtoy


    # println("herew")
    m = allm[k]
    h = allh[k]
    Δ = 0.
    iinfo = -1
    for i=1:N
        mhw = allmhcavtow[k][i]
        mcav = allmcav[k]
        h[i] = sum(mhw) + r*h[i]
        h[i] += h[i] == 0 ? rand([-1,1]) : 0
        oldm = m[i]
        m[i] = h[i]
        for a=1:M
            mcav[a][i] = h[i]-mhw[a]
            # mcav[a][i] = h[i]
            # a == 3 && mcav[a][i] != m[i] && println("%%%%%%%%%%%% i=$i mcav=$(mcav[a][i]) m=$(m[i])")
        end
        if iinfo == i
            println("h[i]=$(h[i]) hold[i]=$(m[i])")
            println("mhw =$(mhw)")
        end
        Δ = max(Δ, abs(m[i] - oldm))
    end
    # println("therew")

    return Δ
end

function updateVarY!(layer::MaxSumLayer, a::Int)
    @extract layer K N M allm allmy allmh B Bup allhy βms
    @extract layer bottom_layer top_layer
    @extract layer allmcav allmycav allmhcavtow allmhcavtoy

    @assert !isbottomlayer(layer)
    #TODO check βms
    # println("here")
    my = allmy[a]
    hy = allhy[a]
    for i=1:N
        mhy = allmhcavtoy[a][i]
        mycav = allmycav[a]

        hy[i] = sum(mhy)
        # isfinite(hy[i])
        B[i,a] = βms*hy[i]
        # pinned from below (e.g. from input layer)
        pu = bottom_layer.Bup[i,a];
        hy[i] += pu/βms
        # !isfinite(hy[i]) && (hy[i] = sign(hy[i]) * ∞ )
        hy[i] = round(Int, hy[i])
        hy[i] += hy[i] == 0 ? rand([-1,1]) : 0
        my[i] = hy[i]
        for k=1:K
            # mycav[k][i] = hy[i]-mhy[k] #TODO
            mycav[k][i] = hy[i]
        end
    end
    # println("there")
end


Θ1(x) = (x-1)*ifelse(x>1,0,1)

function updateFact!(layer::MaxSumLayer, k::Int)
    @extract layer: K N M allm allmy allmh B Bup βms
    @extract layer: bottom_layer top_layer
    @extract layer: allmcav allmycav allmhcavtow allmhcavtoy

    mh = allmh[k];
    pdtop = top_layer.B[k,:]
    ϕ = zeros(F, N)
    iinfo=1; ainfo = -3
    for a=1:M
        mycav = allmycav[a][k]
        mcav = allmcav[k][a]
        mhw = allmhcavtow[k]
        mhy = allmhcavtoy[a]
        ϕy = pdtop[a]
        !isfinite(ϕy) && (ϕy = sign(ϕy) * ∞)
        ϕy = round(Int, ϕy)
        ϕy == 0 && (ϕy += pdtop[a] > 0.5 ? 1 : -1)
        for i=1:N
            ϕ[i] = 0.5*(abs(mcav[i] + mycav[i])-abs(mcav[i] - mycav[i]))
        end

        π = sortperm(ϕ)
        ϕ .= ϕ[π]
        km = searchsortedlast(ϕ,-1.)+1
        # km = searchsortedfirst(ϕ,-1.)
        k0m = searchsortedfirst(ϕ,0.)
        k0p = searchsortedlast(ϕ,0.)
        kp = searchsortedfirst(ϕ,1.)-1
        # kp = searchsortedlast(ϕ,1.)

        Mp = Mm = 0
        Mp -= length(1:km-1)
        Mp += length(kp+1:N)
        Mm = -Mp
        Mp += length(k0m:kp)
        Mm += length(km:k0p)
        np = length(km:k0m-1)
        nm = length(k0p+1:kp)

        xp = round(Int, max(0, ceil((1+np-Mp)/2)))
        xm = round(Int, max(0, ceil((1+nm-Mm)/2)))
        xp = min(xp, np)
        xm = min(xm, nm)

        S1p = sum(ϕ[km:k0m-1])
        S2p = sum(ϕ[k0m-xp:k0m-1])
        S1m = sum(ϕ[k0p+1:k0p])
        S2m = sum(ϕ[k0p+1:k0p+xm])
        Ep = Θ1(Mp - np + 2*xp)  + 2*S2p
        Em = Θ1(Mm - nm + 2*xm)  - 2*S2m
        ϕup = 0.5*(Ep-Em)
        Bup[k,a] = βms*ϕup

        ϕtot = 0.5*(Ep-Em + 2ϕy)

        if a==ainfo
            println("@@@@@@@@@@@@@@@@@@@@@@@@@@")
            println("ϕy = $ϕy")
            println("mcav = $mcav")
            println("mycav = $mycav")
            println("ϕorig = $(ϕ[invperm(π)])")
            println("π = $π")
            println("ϕ = $ϕ")
            println("ks : $km $k0m $k0p $kp")
            println("M : $Mp $Mm")
            println("n : $np $nm")
            println("x : $xp $xm")
            println("E : $Ep $Em")

            println("@@@@@@@@@@@@@@@@@@@@@@@@@@")
        end

        for i=1:N
            j = π[i]
            ϕj = ϕ[j]
            ϕ[j] = 0
            Ecav = zeros(F, 2)
            for η in [-1,1]
                if j < km
                    i==iinfo && a==ainfo && println("case 1")
                    Mpcav = Mp + (η > 0 ? +2 : 0)
                    Mmcav = Mm + (η > 0 ? -2 : 0)
                    npcav = np
                    nmcav = nm
                    ϕp = reverse(ϕ[km:k0m-1])
                    ϕm = ϕ[k0p+1:kp]

                elseif km <= j < k0m
                    i==iinfo && a==ainfo && println("case 2")
                    Mpcav = Mp + (η > 0 ? 1 : -1)
                    Mmcav = Mm + (η > 0 ? -1 : 0)
                    npcav = np + -1
                    nmcav = nm
                    ϕp = ϕ[reverse!([km:j-1; j+1:k0m-1])]
                    ϕm = ϕ[k0p+1:kp]

                elseif k0m <= j <= k0p
                    i==iinfo && a==ainfo && println("case 3")
                    Mpcav = Mp + (η > 0 ? -2 : 0)
                    Mmcav = Mm + (η > 0 ? 0 : -2)
                    npcav = np
                    nmcav = nm
                    ϕp = reverse!(ϕ[km:k0m-1])
                    ϕm = ϕ[k0p+1:kp]
                elseif k0p < j <= kp
                    i==iinfo && a==ainfo && println("case 4 $k0p $j $kp")
                    Mpcav = Mp + (η > 0 ? 0 : -1)
                    Mmcav = Mm + (η > 0 ? -1 : 1)
                    npcav = np
                    nmcav = nm -1
                    ϕp = reverse!(ϕ[km:k0m-1])
                    ϕm = ϕ[reverse!([k0p+1:j-1; j+1:kp])]
                else # j > kp
                    i==iinfo && a==ainfo && println("case 5")
                    Mpcav = Mp + (η > 0 ? 0 : -2)
                    Mmcav = Mm + (η > 0 ? 0 : +2)
                    npcav = np
                    nmcav = nm
                    ϕp = reverse!(ϕ[km:k0m-1])
                    ϕm = ϕ[k0p+1:kp]
                end

                npcav < 0 && (npcav=0)
                nmcav < 0 && (nmcav=0)

                xp = round(Int, max(0, ceil((1+npcav-Mpcav)/2)))
                xm = round(Int, max(0, ceil((1+nmcav-Mmcav)/2)))
                xp = min(xp, npcav)
                xm = min(xm, nmcav)
                if a==ainfo && i==iinfo
                    println("@################")
                    println("η=$η i=$i j=$j")
                    println("ϕy = $ϕy")
                    println("mcav = $mcav")
                    println("mycav = $mycav")
                    println("π = $π")
                    println("ϕ = $ϕ")
                    println("ks : $km $k0m $k0p $kp")
                    println("M : $Mp $Mm")
                    println("Mcav : $Mpcav $Mmcav")
                    println("n : $np $nm")
                    println("ncav : $npcav $nmcav")
                    println("x : $xp $xm")
                end

                S2p = sum(ϕp[1:xp])
                S2m = sum(ϕm[1:xm])

                Epcav = Θ1(Mpcav - npcav + 2*xp) + 2*S2p + ϕy
                Emcav = Θ1(Mmcav - nmcav + 2*xm) - 2*S2m - ϕy
                Ecav[div(3+η,2)] = max(Epcav, Emcav)

                if a==ainfo && i==iinfo
                    println("E : $Ep $Em")
                    println("Ecav-ϕy : $(Epcav-ϕy) $(Emcav+ϕy)")
                    println("S2 : $S2p $S2m")
                    println("####################")
                end


            end

            ϕcav = 0.5*(Ecav[2]-Ecav[1])
            ϕ[j] = ϕj

            mhw[i][a] = 0.5*(abs(ϕcav + mycav[i])-abs(ϕcav - mycav[i]))
            mhy[i][k] = 0.5*(abs(ϕcav + mcav[i])-abs(ϕcav - mcav[i]))
            if a==ainfo && i==iinfo
                println(Ecav[2]," ",Ecav[1])
                println("ϕcav = $ϕcav")
                println("mhw[i][a] $(mhw[i][a])")
                println("mhy[i][k] $(mhy[i][k])")
            end
            # ainfo > 0 && iinfo > 0 &&println("END FACT a=$a i=$i")
        end
    end
end


function update!(layer::MaxSumLayer, reinfpar; mode=:both)
    @extract layer K N M allm allmy allmh B Bup
    # println(layer)
    for k=1:K
        updateFact!(layer, k)
    end
    Δ = 0.
    if !isfrozen(layer)
        for k=1:K
            δ = updateVarW!(layer, k, reinfpar.r)
            Δ = max(δ, Δ)
        end
    end

    if !isbottomlayer(layer)
        for a=1:M
            updateVarY!(layer, a)
        end
    end

    return Δ
end

function initrand!(layer::MaxSumLayer)
    @extract layer: K N M allm allmy allmh B Bup 
    @extract layer: allmcav allmycav allmhcavtow allmhcavtoy

    for m in allm
        m .= rand([-1,1], N)
    end
    for my in allmy
        my .= rand([-1,1], N)
    end
    for mh in allmh
        mh .= rand([-1,1], M)
    end

    for k=1:K,a=1:M,i=1:N
        allmcav[k][a][i] = allm[k][i]
        # allmycav[a][k][i] = allmy[a][i]
        allmhcavtow[k][i][a] = allmh[k][a]*allmy[a][i]
        allmhcavtoy[a][i][k] = allmh[k][a]*allm[k][i]
    end
end

function fixY!(layer::MaxSumLayer, x::AbstractMatrix)
    @extract layer: K N M allm allmy allmh B Bup
    @extract layer: allhy allmycav

    @assert all(x -> x==1 || x==-1, x) # works only on binary input
    
    for a=1:M, i=1:N
        allhy[a][i] = sign(x[i, a]) * 100
        # allmy[a][i] = x[i,a]
        allmy[a][i] = allhy[a][i]
        mycav = allmycav[a]
        for k=1:K
            mycav[k][i] = allhy[a][i]
        end
    end
end

function getW(layer::L) where L <: Union{MaxSumLayer}
    @extract layer: allm K
    # TODO add weight_mask
    return vcat([(sign.(allm[k] .+ 1f-10))' for k in 1:K]...)
end

function forward(layer::L, x) where L <: Union{MaxSumLayer}
    @extract layer: N K
    @assert size(x, 1) == N
    W = getW(layer)
    @assert size(W) == (K, N)
    return sign.(W*x .+ 1f-10)
end
