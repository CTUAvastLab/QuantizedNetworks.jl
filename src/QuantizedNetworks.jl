module QuantizedNetworks

using Distributions
using ChainRulesCore
using Flux
using NNlib
using Zygote

using Base: RefValue
using ChainRulesCore: rrule, @scalar_rule, NoTangent, ProjectTo
using Flux: glorot_uniform, @functor, _create_bias
using NNlib: hardtanh

export ClippedArray, L0Gate
export AbstractEstimator, STE, PolynomialSTE, SwishSTE
export AbstractQuantizer, Sign, Heaviside, Ternary
export QuantDense, FQuantizer, FQuantizer2

include("clippedarray.jl")
include("l0gate.jl")
include("estimators.jl")
include("quantizers.jl")
include("layers/quantdense.jl")
include("layers/fquantizer.jl")

end # module QuantizedNetworks
