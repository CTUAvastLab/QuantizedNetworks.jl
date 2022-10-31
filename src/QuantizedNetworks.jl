module QuantizedNetworks

using Distributions
using ChainRulesCore
using Flux
using NNlib
using Zygote

using Base: RefValue
using ChainRulesCore: rrule, @scalar_rule, NoTangent, ProjectTo
using Flux: glorot_uniform, @functor, _create_bias
using Flux.Optimise: AbstractOptimiser, apply!
using NNlib: hardtanh

export ClippedArray, L0Gate
export AbstractEstimator, STE, PolynomialSTE, SwishSTE
export AbstractQuantizer, Sign, Heaviside, Ternary
export QuantDense, FQuantizer, FeatureQuantizer, MissingQuantizer
export Bop, CaseOptimizer, isbinary

include("utilities.jl")
include("clippedarray.jl")
include("l0gate.jl")
include("estimators.jl")
include("quantizers.jl")
include("bop.jl")

include("layers/quantdense.jl")
include("layers/fquantizer.jl")
include("layers/featurequantizer.jl")
include("layers/missingquantizer.jl")

end # module QuantizedNetworks
