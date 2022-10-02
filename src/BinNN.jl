module BinNN

using Distributions
using ChainRulesCore
using Flux
using NNlib

using Base.Broadcast: broadcasted
using ChainRulesCore: rrule, @scalar_rule, NoTangent, ProjectTo
using Flux: glorot_uniform, @functor, BatchNorm
using NNlib: hardtanh

export binarize, binarize_stochastic
export ClippedArray, BinDense

const Numeric = Union{AbstractArray{<:T},T} where {T<:Number}

include("binarization.jl")
include("clippedarray.jl")
include("layers/bindense.jl")

end # module BinNN
