cpu(x) = fmap(x -> adapt(Array, x), x) 

gpu(x) = fmap(CUDA.cu, x) 
# CUDA.cu(x::Integer) = x
CUDA.cu(x::Float64) = Float32(x)
CUDA.cu(x::Array{Int64}) = convert(CuArray{Int32}, x)
