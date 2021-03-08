cpu(x) = fmap(x -> adapt(Array, x), x) 

gpu(x) = fmap(CUDA.cu, x) 
# CUDA.cu(x::Integer) = x
CUDA.cu(x::Float64) = Float32(x)
CUDA.cu(x::Array{Int64}) = convert(CuArray{Int32}, x)

"""
    @gpu f(x) = ...

Define a function to be broadcastable over `CuArray`s,
e.g. `f.(CUDA.zeros(2))`.
"""
macro gpu(ex)
    quote
        $(esc(ex))  # define cpu version
        CUDA.@cufunc $ex     # define gpu version
    end
end