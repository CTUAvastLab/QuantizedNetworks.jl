using Test
using QuantizedNetworks
using ChainRulesCore
using ChainRulesTestUtils
using Zygote


@testset "clippedarray" begin include("clippedarray.jl"); end
@testset "estimators" begin include("estimators.jl"); end
@testset "quantizers" begin include("quantizers.jl"); end

@testset "layers/quantdense" begin include("layers/quantdense.jl"); end
@testset "layers/fquantizer" begin include("layers/featurequantizer.jl"); end
