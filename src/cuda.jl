
gpu(x::Array) = CUDA.cu(x)
gpu(x::CUDA.CuArray) = x

# go one level deep
function gpu(x::T) where T
    cufields = [CUDA.cu(getfield(x, f)) for f in fieldnames(T)]
    T(cufields...)
end

cpu(x) = x

"""
    @gpu f(x) = ...

Define a function to be broadcstable over `CuArray`s,
e.g. `f.(CUDA.zeros(2))`.
"""
macro gpu(ex)
    quote
        $(esc(ex))  # define cpu version
        CUDA.@cufunc $ex     # define gpu version
    end
end