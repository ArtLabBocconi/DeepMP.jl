G(x) = exp(-(x^2)/2) / √(2f0*π)
H(x) = erfc(x / √2f0) / 2
CUDA.@cufunc H(x) = CUDA.erfc(x / √2f0) / 2

lg2 = log(2f0)

∞atanh = 25.

function myatanh(x)
    y = atanh(x)
    return isfinite(y) ? y : sign(x)*∞atanh
end

myatanh(p,m) = _myatanh(p/(p+m), m/(p+m))
function _myatanh(p,m)
    # @assert p >= 0 "p >= 0 p=$p"
    # @assert m >= 0
    p == 0 && return -∞atanh
    m == 0 && return ∞atanh
    y = 0.
    if m < 1f-10
        y = 0.5*(lg2 - log(2m))
    elseif p < 1f-10
        y = -0.5*(lg2 - log(2p))
    else
        y = atanh(p-m)
    end
    @assert isfinite(y) "y=$y p=$p m=$m"
    return y
end

"""
Numerically stable log(1 + exp(x))
"""
function log1pexp(x)
    m = max(0, x) 
    m + log1p(exp(-abs(x)))
end

logcosh(x) = abs(x) > 600 ? abs(x) - lg2 : log(cosh(x))
logsinhabs(x) = abs(x) > 600 ? abs(x) - lg2 : log(sinh(abs(x)))
CUDA.@cufunc logcosh(x) = abs(x) > 600 ? abs(x) - lg2 : log(cosh(x))
CUDA.@cufunc logsinhabs(x) = abs(x) > 600 ? abs(x) - lg2 : log(sinh(abs(x)))

atanh2Hm1(x) = abs(x) > 6 ? -sign(x)*0.25*(log(2π) + x^2 + 2log(abs(x))) :
                atanh(2H(x)-1)

function GHapp(x)
    y = 1/x
    y2 = y^2
    x + y * (1 - 2y2 * (1 - 5y2 * (1 - 7.4f0y2)))
end

CUDA.@cufunc function GHapp(x)
    y = 1/x
    y2 = y^2
    x + y * (1 - 2y2 * (1 - 5y2 * (1 - 7.4f0y2)))
end

GH(x) = x > 30 ? GHapp(x) : G(x) / H(x)
_GH(x) = x > 30 ? GHapp(x) : G(x) / H(x)
CUDA.@cufunc _GH(x) = x > 30 ? GHapp(x) : G(x) / H(x)

function GHnaive(uσ, x)
    Hp = H(x)
    Hm = 1-Hp
    Gp = G(x)
    p = (tanh(uσ)+1)/2
    Gp*(2p-1) / (p*Hp + (1-p)*Hm)
end

CUDA.@cufunc function GHnaive(uσ, x)
    Hp = H(x)
    Hm = 1-Hp
    Gp = G(x)
    p = (tanh(uσ)+1)/2
    Gp*(2p-1) / (p*Hp + (1-p)*Hm)
end

function GH(uσ, x)
    uσ == 0 && return zero(x)
    uσ == Inf && return GH(x)
    uσ == -Inf && return -GH(-x)
    abs(x) < 5 && return GHnaive(uσ, x)
    uh = atanh2Hm1(x)
    ex = (logsinhabs(uσ) + logcosh(uh)) - (logcosh(uσ+uh) + x^2/2)
    if abs(ex) > 600
        ex = sign(ex) * 600
    end
    res = sign(uσ)* exp(ex) * √(2f0/π)
    # if !isfinite(res)
    #     @show p up ug uh ex log(abs(mp)) logcosh(up)  logcosh(uh) logcosh(up+uh)
    # end
    # @assert isfinite(res)
    return res
end

CUDA.@cufunc function GH(uσ, x)
    uσ == 0 && return zero(x)
    uσ == Inf && return _GH(x)
    uσ == -Inf && return -_GH(-x)
    return GHnaive(uσ, x)
    # abs(x) < 5 && return GHnaive(uσ, x)
    # uh = atanh2Hm1(x)
    # ex = (logsinhabs(uσ) + logcosh(uh)) - (logcosh(uσ+uh) + x^2/2)
    # # @show x mσ
    # # ex = (logsinhabs(uσ) + logcosh(uh)) - (logcosh(uσ+uh) + x^2/2)
    # if abs(ex) > 600
    #     ex = sign(ex) * 600
    # end
    # res = sign(uσ)* exp(ex) * √(2f0/π)
    # # if !isfinite(res)
    # #     @show p up ug uh ex log(abs(mp)) logcosh(up)  logcosh(uh) logcosh(up+uh)
    # # end
    # # @assert isfinite(res)
    # return res
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
