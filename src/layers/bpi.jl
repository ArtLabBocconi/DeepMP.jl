
###########################
#       BPI LAYER
#######################################
mutable struct BPILayer <: AbstractLayer
    l::Int

    K::Int
    N::Int
    M::Int

    allm::VecVec
    allmy::VecVec

    allh::VecVec # for W reinforcement
    allhext::VecVec # for W reinforcement
    allhy::VecVec # for Y reinforcement

    allpu::VecVec # p(σ=up) from fact ↑ to y
    allpd::VecVec # p(σ=up) from y  ↓ to fact

    Mtot::VecVec
    MYtot::VecVec

    top_allpd::VecVec
    bottom_allpu::VecVec

    top_layer::AbstractLayer
    bottom_layer::AbstractLayer

    weight_mask::Vector{Vector{Int}}
    isfrozen::Bool
end

function BPILayer(K::Int, N::Int, M::Int; density=1, isfrozen=false)
    # for variables W
    allm = [zeros(N) for i=1:K]
    allh = [zeros(N) for i=1:K]
    allhext = [zeros(N) for i=1:K]
    Mtot = [zeros(N) for i=1:K]

    # for variables Y
    allmy = [zeros(N) for a=1:M]
    allhy = [zeros(N) for a=1:M]
    MYtot = [zeros(N) for a=1:M]


    allpu = [zeros(M) for k=1:K]
    allpd = [zeros(M) for k=1:N]

    weight_mask = [[rand() < density ? 1 : 0 for i=1:N] for i=1:K]

    return BPILayer(-1, K, N, M
        , allm, allmy
        , allh, allhext, allhy, allpu,allpd
        , Mtot, MYtot, VecVec(), VecVec()
        , DummyLayer(), DummyLayer()
        , weight_mask, isfrozen)
end

function updateFact!(layer::BPILayer, k::Int, reinfpar)
    @extract layer: K N M allm allmy allpu allpd
    @extract layer: MYtot Mtot bottom_allpu top_allpd

    m = allm[k]
    Mt = Mtot[k]
    pd = top_allpd[k]
    mask = layer.weight_mask[k]
    for a=1:M
        my = allmy[a]
        MYt = MYtot[a]
        Mhtot = 0.
        Chtot = 1e-10
        if !isbottomlayer(layer)
            for i=1:N
                Mhtot += my[i]*m[i] * mask[i]
                Chtot += (1 - my[i]^2*m[i]^2) * mask[i]
            end
        else
            for i=1:N
                Mhtot += my[i]*m[i] * mask[i]
                Chtot += (my[i]^2 *(1 - m[i]^2)) * mask[i]
            end
        end

        for i=1:N
            mask[i] == 1 || continue
            mh = 1/√Chtot * GH(pd[a], -(Mhtot - my[i] * m[i]) / √Chtot)
            Mt[i] += my[i] * mh
            if !isbottomlayer(layer)
                MYt[i] += m[i] * mh
            end
        end

        # Message to top
        allpu[k][a] = atanh2Hm1(-Mhtot / √Chtot)
    end
end

function updateVarW!(layer::L, k::Int, r::Float64=0.) where {L <: Union{BPILayer}}
    @extract layer: K N M allm allmy  allpu allpd l
    @extract layer: MYtot Mtot  bottom_allpu allh allhext
    Δ = 0.
    m = allm[k]
    Mt = Mtot[k]
    h = allh[k]
    hext = allhext[k]

    for i=1:N
        if layer.weight_mask[k][i] == 0
            @assert m[i] == 0 "m[i]=$(m[i]) shiuld be 0"
        end
        h[i] = Mt[i] + r*h[i] + hext[i]
        oldm = m[i]
        m[i] = tanh(h[i])
        Δ = max(Δ, abs(m[i] - oldm))
    end
    return Δ
end

function updateVarY!(layer::L, a::Int, ry::Float64=0.) where {L <: Union{BPILayer}}
    @extract layer K N M allm allmy allpu allpd
    @extract layer allhy MYtot Mtot  bottom_allpu

    @assert !isbottomlayer(layer)

    MYt = MYtot[a]
    my = allmy[a]
    hy = allhy[a]

    for i=1:N
        hy[i] = MYt[i] + ry* hy[i]
        allpd[i][a] = hy[i]
        pu = bottom_allpu[i][a];
        hy[i] += pu
        my[i] = tanh(hy[i])
    end
end

function initYBottom!(layer::L, a::Int, ry::Float64=0.) where {L <: Union{BPILayer}}
    @extract layer: K N M allm allmy allpu allpd
    @extract layer: allhy MYtot Mtot bottom_allpu

    @assert isbottomlayer(layer)

    x = layer.bottom_layer.x
    my = allmy[a]
    for i=1:N
        my[i] = x[i, a]
    end
end

function update!(layer::L, reinfpar) where {L <: Union{BPILayer}}
    @extract layer: K N M allm allmy allpu allpd  MYtot Mtot


    #### Reset Total Fields
    for a=1:M
        MYtot[a] .= 0
    end
    for k=1:K
        Mtot[k] .= 0
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

    # bypass Y if toplayer
    if !isbottomlayer(layer)
        for a=1:M
            updateVarY!(layer, a, reinfpar.ry)
        end
    end
    return Δ
end

function initrand!(layer::L) where {L <: Union{BPILayer}}
    @extract layer K N M allm allmy allpu allpd  top_allpd
    ϵ = 1e-1
    mask = layer.weight_mask

    for (k, m) in enumerate(allm)
        m .= (2*rand(N) .- 1) .* ϵ .* mask[k]
    end

    for my in allmy
        my .= (2*rand(N) .- 1) .* ϵ
    end

    for pu in allpu
        pu .= rand(M)
    end

    for pd in allpd
        pd .= rand(M)
    end
end

function fixW!(layer::L, w=1.) where {L <: Union{BPILayer}}
    @extract layer: K N M allm allmy allpu allpd  top_allpd

    for k=1:K, i=1:N
        allm[k][i] = w
    end
end

function fixY!(layer::L, x::Matrix) where {L <: Union{BPILayer}}
    @extract layer K N M allm allmy allpu allpd  top_allpd

    for a=1:M, i=1:N
        allmy[a][i] = x[i,a]
    end
end

function getW(layer::L) where L <: Union{BPILayer}
    @extract layer: weight_mask allm K
    return vcat([(sign.(allm[k] .+ 1e-10) .* weight_mask[k])' for k in 1:K]...)
end

function forward(layer::L, x) where L <: Union{BPILayer}
    @extract layer: N K
    @assert size(x, 1) == N
    W = getW(layer)
    @assert size(W) == (K, N)
    return sign.(W*x .+ 1e-10)
end
