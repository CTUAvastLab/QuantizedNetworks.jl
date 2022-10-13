module QuantizedNetworks

using Distributions
using ChainRulesCore
using Flux
using NNlib

using ChainRulesCore: rrule, @scalar_rule, NoTangent, ProjectTo
using Flux: glorot_uniform, @functor, _create_bias
using NNlib: hardtanh

export ClippedArray
export AbstractEstimator, STE, PolynomialSTE, SwishSTE
export AbstractQuantizer, Sign, Heaviside, Ternary
export QuantDense, FQuantizer

include("clippedarray.jl")
include("estimators.jl")
include("quantizers.jl")
include("layers/quantdense.jl")
include("layers/fquantizer.jl")

end # module QuantizedNetworks
