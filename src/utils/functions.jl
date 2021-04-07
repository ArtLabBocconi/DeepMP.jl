G(x::T) where T = exp(-(x^2) / 2) / √(T(2)*π)
H(x::T) where T = erfc(x / √T(2)) / 2

∞atanh = 25.0

function myatanh(x)
    y = atanh(x)
    return isfinite(y) ? y : sign(x)*∞atanh
end

myatanh(p,m) = _myatanh(p/(p+m), m/(p+m))

function _myatanh(p::T, m::T) where T
    # @assert p >= 0 "p >= 0 p=$p"
    # @assert m >= 0
    p == 0 && return -T(∞atanh)
    m == 0 && return T(∞atanh)
    if m < 1f-10
        y = (log(T(2)) - log(2m)) / 2
    elseif p < 1f-10
        y = -(log(T(2)) - log(2p)) / 2
    else
        y = atanh(p-m)
    end
    @assert isfinite(y) "y=$y p=$p m=$m"
    return y
end

function logcosh(x::T) where T
    ax = abs(x)
    return ax + log1p(exp(-2ax)) - log(T(2))
end

function logsinhabs(x::T) where T
    ax = abs(x)
    return ax + log1p(-exp(-2ax)) - log(T(2))
end

function atanherf(x)
    ax = abs(x)
    #logerfc(ax) = log(erfcx(ax)) - ax^2 in SpecialFunctions.jl
    return sign(x) * (log(2) + log1p(-erfc(ax)/2) - log(erfcx(ax)) + ax^2) / 2
end

atanh2Hm1(x::T) where T = -atanherf(x / √T(2))

GH(x::T) where T = √(T(2) / π) / erfcx(x / √T(2))

function GHt(m, x::T) where T
    r = GH(x)
    if m ≠ 1
        f = erfc(x / √T(2))
        r *= m * (f / (1 - m + m * f))
    end
    return r
end

function GH2(uσ, x)
    return GHt(tanh(uσ), x)
end

# TODO approx
function DH(σu, x, y, C)
    p = (1+tanh(σu)) /2
    Hpp = H(-(x+y)/C)
    Hpm = H(-(x-y)/C)
    Hmp = 1 - Hpp
    Hmm = 1 - Hpm
    (p*(Hpp - Hpm) + (1-p)*(Hmp - Hmm)) / (p*(Hpp + Hpm) + (1-p)*(Hmp + Hmm))
end
