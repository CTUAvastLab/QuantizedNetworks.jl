quantizers_sign = [
    Sign(),
    Sign(STE()),
    Sign(STE(1)),
    Sign(STE(2)),
    Sign(PolynomialSTE()),
    Sign(SwishSTE()),
    Sign(SwishSTE(5)),
    Sign(SwishSTE(10)),
]

quantizers_heaviside = [
    Heaviside(),
    Heaviside(STE()),
    Heaviside(STE(1)),
    Heaviside(STE(2)),
]

quantizers_ternary = [
    Ternary(),
    Ternary(0.05, STE()),
    Ternary(0.05, STE(1)),
    Ternary(0.05, STE(2)),
]

quantizers = [
    identity,
    quantizers_sign...,
    quantizers_heaviside...,
    quantizers_ternary...,
]

activations = [
    identity,
    tanh,
]

@testset "Default constructors" begin
    d1 = QuantizedDense(10 => 100)
    d2 = QuantizedDense(d1.weight)
    d3 = QuantizedDense(d1.weight, false)
    d4 = QuantizedDense(d1.weight, false, identity)
    d5 = QuantizedDense(d1.weight, false, identity, Ternary())
    for l in [d1, d2, d3, d4]
        @test isa(l, QuantizedDense)
        @test isa(l.weight, ClippedArray)
        @test size(l.weight) == (100, 10)
        @test l.bias == false
        @test l.σ == identity
        @test l.quantizer == Ternary()
    end

    l1 = QuantizedDense(10 => 100; bias = true)
    l2 = QuantizedDense(l1.weight, l1.bias)
    l3 = QuantizedDense(l1.weight, l1.bias, identity)
    l4 = QuantizedDense(l1.weight, l1.bias, identity, Ternary())
    for l in [l1, l2, l3, l4]
        @test isa(l, QuantizedDense)
        @test isa(l.weight, ClippedArray)
        @test size(l.weight) == (100, 10)
        @test size(l.bias) == (100, )
        @test l.σ == identity
        @test l.quantizer == Ternary()
    end

    @test_throws MethodError QuantizedDense(10 => 10.5)
    @test_throws MethodError QuantizedDense(10 => 10.5, tanh)
    @test_throws DimensionMismatch QuantizedDense(3 => 4; bias=rand(5))
    @test_throws DimensionMismatch QuantizedDense(rand(4, 3), rand(5))
    @test_throws MethodError QuantizedDense(rand(5))
    @test_throws MethodError QuantizedDense(rand(5), rand(5))
    @test_throws MethodError QuantizedDense(rand(5), rand(5), tanh)
end

@testset "quantizer = $(q)" for q in quantizers
    @testset "activation = $(σ)" for σ in activations
        @testset "constructors" begin
            l1 = QuantizedDense(10 => 100, σ; quantizer = q, bias = false)
            l2 = QuantizedDense(l1.weight, l1.bias, l1.σ, q)
            for l in [l1, l2]
                @test isa(l, QuantizedDense)
                @test isa(l.weight, ClippedArray)
                @test size(l.weight) == (100, 10)
                @test l.bias == false
                @test l.σ == σ
                @test l.quantizer == q
            end

            l3 = QuantizedDense(10 => 100, σ; quantizer = q, bias = true)
            l4 = QuantizedDense(l3.weight, l3.bias, l3.σ, q)
            for l in [l3, l4]
                @test isa(l, QuantizedDense)
                @test isa(l.weight, ClippedArray)
                @test size(l.weight) == (100, 10)
                @test size(l.bias) == (100, )
                @test l.σ == σ
                @test l.quantizer == q
            end
        end
        @testset "dimensions" begin
            l1 = QuantizedDense(10 => 5, σ; quantizer = q, bias = false)
            l2 = QuantizedDense(10 => 5, σ; quantizer = q, bias = true)

            for l in [l1, l2]
                @test length(l(randn(10))) == 5
                @test_throws DimensionMismatch l(randn(2))
                @test_throws MethodError l(1) # avoid broadcasting
                @test_throws MethodError l.(randn(10)) # avoid broadcasting
                @test size(l(randn(10))) == (5,)
                @test size(l(randn(10, 2))) == (5, 2)
                @test size(l(randn(10, 2, 3))) == (5, 2, 3)
                @test size(l(randn(10, 2, 3, 4))) == (5, 2, 3, 4)
                @test_throws DimensionMismatch l(randn(11, 2, 3))
            end
        end
    end
end

weight = Float64[
    -2  -1  0  1  2
    -4  -2  0  2  4
]

input1 = Float64[1, 4, 3, 2, 5]
input2 = hcat(input1, input1)

@testset "outputs" begin
    @testset "quantizer = $(q)" for q in quantizers_sign
        l = QuantizedDense(weight, false, identity, q)
        @test l(input1) == 5 .* ones(2)
        @test l(input2) == 5 .* ones(2, 2)
    end
    @testset "quantizer = $(q)" for q in quantizers_heaviside
        l = QuantizedDense(weight, false, identity, q)
        @test l(input1) == 7 .* ones(2)
        @test l(input2) == 7 .* ones(2, 2)
    end
    @testset "quantizer = $(q)" for q in quantizers_ternary
        l = QuantizedDense(weight, false, identity, q)
        @test l(input1) == 2 .* ones(2)
        @test l(input2) == 2 .* ones(2, 2)
    end
end
