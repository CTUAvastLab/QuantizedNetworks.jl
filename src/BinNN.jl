module BinNN

using Distributions
using ChainRulesCore
using Flux
using NNlib

using ChainRulesCore: rrule, @scalar_rule, NoTangent, ProjectTo
using Flux: glorot_uniform, @functor, _create_bias
using NNlib: hardtanh
using Flux.Optimise: AbstractOptimiser

export ClippedArray
export value, deriv
export AbstractEstimator, STE, PolynomialSTE, SwishSTE
export AbstractQuantizer, Sign, Heaviside, Ternary
export QuantDense
export Bop

include("clippedarray.jl")
include("estimators.jl")
include("quantizers.jl")
include("layers/quantdense.jl")
include("optimisers/bop.jl")

end # module BinNN
