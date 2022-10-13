using Test
using QuantizedNetworks

using QuantizedNetworks.ChainRulesCore
using QuantizedNetworks.Flux.Zygote

@testset "clippedarray" begin include("clippedarray.jl"); end
@testset "estimators" begin include("estimators.jl"); end
@testset "quantizers" begin include("quantizers.jl"); end

@testset "layers/quantdense" begin include("layers/quantdense.jl"); end
