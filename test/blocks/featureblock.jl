using QuantizedNetworks: _forward_pass

quantizers = [
    identity,
    Sign(),
    Sign(STE()),
    Sign(STE(1)),
    Sign(STE(2)),
    Sign(PolynomialSTE()),
    Sign(SwishSTE()),
    Sign(SwishSTE(5)),
    Sign(SwishSTE(10)),
    Heaviside(),
    Heaviside(STE()),
    Heaviside(STE(1)),
    Heaviside(STE(2)),
    Ternary(),
    Ternary(0.05, STE()),
    Ternary(0.05, STE(1)),
    Ternary(0.05, STE(2)),
]

inputs = [
    rand(7, 3),
    hcat(randn(2, 3), [1,missing]),
]

ks = 1:2:10

@testset "quantizer = $(quantizer)" for quantizer in quantizers
    @testset "input = $(input)" for input in inputs
        @testset "T = $(T)" for T in [Float32, Float64]
            contains_missing = any(ismissing, input)
            if contains_missing && (quantizer == identity)
                continue
            end
            if contains_missing
                x = convert(Matrix{Union{T, Missing}}, input)
            else
                x = T.(input)
            end

            @testset "k = $k" for k in ks
                q1 = FeatureBlock(size(x, 1), k; quantizer, output_missing = false)
                q2 = FeatureBlock(size(x, 1), k; quantizer, output_missing = true)

                @testset "Layers" begin
                    @test isa(q1.layers, Parallel)
                    @test isa(q1[1], FeatureQuantizer)
                    @test isa(q1.layers[1], FeatureQuantizer)

                    @test isa(q2.layers, Parallel)
                    @test isa(q2[1], FeatureQuantizer)
                    @test isa(q2.layers[1], FeatureQuantizer)
                    @test isa(q2[2], typeof(quantizer))
                    @test isa(q2.layers[2], typeof(quantizer))
                end

                @testset "Output type and size" begin
                    @test isa(q1(x), Matrix{T})
                    @test isa(q2(x), Matrix{T})

                    @test size(q1(x)) == (size(x, 1) * k, size(x, 2))
                    @test size(q1(x)) == size(q2[1](x))
                    @test size(q2(x)) == (size(x, 1) * (k + 1), size(x, 2))
                end

                @testset "Output values" begin
                    @test q1(x) == q1[1](x)
                    @test q2(x) == vcat(q2[1](x), q2[2](x))
                    @test q2[2](x) == quantizer(x)
                    @test q2(x)[(size(x, 1) * k + 1):end, :] == quantizer(x)
                end
            end
        end
    end
end
