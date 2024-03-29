using Test

using QuantizedNetworks
using ChainRulesTestUtils
using FiniteDifferences

using QuantizedNetworks.ChainRulesCore
using QuantizedNetworks.Flux
using QuantizedNetworks.Zygote

@testset "QuantizedNetworks" begin
@testset "clippedarray" begin include("clippedarray.jl"); end
@testset "estimators" begin include("estimators.jl"); end
@testset "quantizers" begin include("quantizers.jl"); end

@testset "layers/quantdense" begin include("layers/quantdense.jl"); end
@testset "layers/fquantizer" begin include("layers/fquantizer.jl"); end
@testset "layers/featurequantizer" begin include("layers/quantizeddense.jl"); end
@testset "layers/featurequantizer" begin include("layers/featurequantizer.jl"); end

@testset "blocks/denseblock" begin include("blocks/denseblock.jl") end
@testset "blocks/denseblock" begin include("blocks/featureblock.jl") end
end
