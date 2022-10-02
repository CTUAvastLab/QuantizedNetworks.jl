module BinNN

using Distributions
using ChainRulesCore
using Flux

using Broadcast: broadcasted
using ChainRulesCore: rrule, @scalar_rule, NoTangent, project_x, ProjectTo
using Flux: glorot_uniform, _create_bias, @functor

export binarize, binarize_stochastic

const Numeric = Union{AbstractArray{<:T},T} where {T<:Number}

include("binarization.jl")
include("clippedarray.jl")

end # module BinNN
