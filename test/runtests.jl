using Test
using BinNN

using BinNN.ChainRulesCore
using BinNN.Flux.Zygote

@testset "clippedarray" begin include("clippedarray.jl"); end
@testset "estimators" begin include("estimators.jl"); end
@testset "quantizers" begin include("quantizers.jl"); end

@testset "layers/quantdense" begin include("layers/quantdense.jl"); end
