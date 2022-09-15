module DeepMP

using ExtractMacro
using DelimitedFiles
using SpecialFunctions
using Printf
using Random
using LinearAlgebra
using Statistics
using Base: @propagate_inbounds # for DataLoader
using Tullio
using LoopVectorization
using CUDA, KernelAbstractions, CUDAKernels
using Adapt
using Functors
using JLD2
import Zygote
import ForwardDiff
using Base: @kwdef

CUDA.allowscalar(false)

# using PyPlot

const F = Float64
const CVec = Vector{Complex{F}}
const IVec = Vector{Int}
const Vec = Vector{F}
const VecVec = Vector{Vec}
const IVecVec = Vector{IVec}
const VecVecVec = Vector{VecVec}
const IVecVecVec = Vector{IVecVec}

include("cuda.jl")
include("utils/utils.jl")
include("utils/functions.jl")
include("utils/dataloader.jl")

include("channels/channels.jl")
include("channels/sign.jl")
include("channels/relu.jl")

include("layers/layers.jl")
include("factor_graph.jl")
include("reinforcement.jl")

include("solve.jl")
# export generate_problem, solve, converge!

end #module
