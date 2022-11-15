@testset "Default constructors" begin
    d1 = DenseBlock(10 => 100)
    d2 = DenseBlock(10 => 100, identity)
    d3 = DenseBlock(10 => 100, identity; weight_quantizer=Ternary())
    d4 = DenseBlock(10 => 100, identity; output_quantizer=Sign())
    d5 = DenseBlock(10 => 100, identity; bias=false)

    for l in [d1, d2, d3, d4, d5]
        @test isa(l, DenseBlock)
        @test isa(l.layers, Chain)
        @test isa(l[1], QuantizedDense)
        @test isa(l[2], BatchNorm)
        @test isa(l[3], Sign)
    end
end

quantizers = [
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

            d = DenseBlock(size(x, 1) => 10; output_quantizer = quantizer)

            @testset "Output type, size and values" begin
                @test isa(d(x), Matrix{T})
                @test size(d(x)) == (10, size(x, 2))
                @test size(d[1](x)) == (10, size(x, 2))
                @test size(d[1:2](x)) == (10, size(x, 2))
                @test size(d[1:3](x)) == (10, size(x, 2))
                @test d(x) == d[3](d[2](d[1](x)))
            end
        end
    end
end
