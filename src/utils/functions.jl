G(x::T) where T = exp(-(x^2) / 2) / √(T(2)*π)
H(x::T) where T = erfc(x / √T(2)) / 2
CUDA.@cufunc H(x::T) where T = CUDA.erfc(x / √T(2)) / 2

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

## Numerically stable log(1 + exp(x))
@gpu function log1pexp(x)
    m = max(zero(x), x) 
    m + log1p(exp(-abs(x)))
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
    ax ≤ 1 && return atanh(erf(x))
    return sign(x) * (log(2) + log1p(-erfc(ax)/2) - logerfc(ax)) / 2
end
atanh2Hm1(x::T) where T = -atanherf(x / √T(2))

# cuda version, any change crashes julia
CUDA.@cufunc atanh2Hm1(x) = atanh(2H(x)-1)
CUDA.@cufunc logcosh(x) = log(cosh(x))
CUDA.@cufunc logsinhabs(x) = log(sinh(abs(x)))

GH(x::T) where T = √(T(2) / π) / erfcx(x / √T(2))

## in case cuda version of GH is needed
# @gpu function GHapp(x)
#     y = 1/x
#     y2 = y^2
#     x + y * (1 - 2y2 * (1 - 5y2 * (1 - 7.4f0y2)))
# end
# CUDA.@cufunc GH(x) = x > 30 ? GHapp(x) : G(x) / H(x)

@gpu function GHt(m, x::T) where T
    r = GH(x)
    if m ≠ 1
        f = erfc(x / √T(2))
        r *= m * (f / (1 - m + m * f))
    end
    return r
end
@gpu function GH2(uσ, x)
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
