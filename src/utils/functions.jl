
@inline Cassette.overdub(::CUDAKernels.CUDACtx, ::typeof(SpecialFunctions.erfcx), x::Union{Float32, Float64}) = SpecialFunctions.erfcx(x)
@inline Cassette.overdub(::KernelAbstractions.CPUCtx, ::typeof(SpecialFunctions.erfcx), x::Union{Float32, Float64}) = SpecialFunctions.erfcx(x)

G(x::T) where T = exp(-(x^2) / 2) / √(T(2)*π)
H(x::T) where T = erfc(x / √T(2)) / 2

logH(x::T) where T = logerfc(x/√T(2)) - log(2)
logG(x::T) where T = -x^2/2 - log(T(2)*π)/2

function logcosh(x::T) where T
    ax = abs(x)
    return ax + log1p(exp(-2ax)) - log(T(2))
end

function logsinhabs(x::T) where T
    ax = abs(x)
    return ax + log1p(-exp(-2ax)) - log(T(2))
end

function atanherf(x::T) where T
    ax = abs(x)
    #logerfc(ax) = log(erfcx(ax)) - ax^2 in SpecialFunctions.jl
    return sign(x) * (log(T(2)) + log1p(-erfc(ax)/2) - log(erfcx(ax)) + ax^2) / 2
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

# V here is √V in the paper
compute_g(B, ω, V) = GH2(B, -ω / V) / V

function GH2(uσ, x)
    m = tanh(uσ)
    s = sign(uσ)
    return sign(m) * GHt(abs(m), s*x)
    # return m > 0 ? GHt(m, x) : - GHt(-m, -x) 
    # return GHt(tanh(uσ), x)
end
