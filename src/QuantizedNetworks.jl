module QuantizedNetworks

using Distributions
using ChainRulesCore
using Flux
using NNlib
using Zygote

using Base: RefValue
using ChainRulesCore: rrule, @scalar_rule, NoTangent, ProjectTo
using Flux: glorot_uniform, @functor, create_bias
using Flux.Optimise: AbstractOptimiser, apply!
using NNlib: hardtanh

export ClippedArray, L0Gate
export AbstractEstimator, STE, PolynomialSTE, SwishSTE, StochasticSTE
export AbstractQuantizer, Sign, Heaviside, Ternary
export QuantDense, FQuantizer
export Bop, CaseOptimizer, isbinary

export QuantizedDense, FeatureQuantizer
export AbstractBlock, DenseBlock, FeatureBlock

include("clippedarray.jl")
include("l0gate.jl")
include("estimators.jl")
include("quantizers.jl")
include("bop.jl")

include("layers/featurequantizer.jl")
include("layers/quantdense.jl")
include("layers/quantizeddense.jl")
include("layers/fquantizer.jl")

abstract type AbstractBlock end

(m::AbstractBlock)(x) = m.layers(x)
Base.getindex(m::AbstractBlock, inds) = m.layers[inds]

include("blocks/denseblock.jl")
include("blocks/featureblock.jl")

end # module QuantizedNetworks
