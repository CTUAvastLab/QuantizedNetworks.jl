using QuantizedNetworks: forward_pass, pullback

# Sign function
@testset "Sing quantizer" begin
    Ts = [Float64, Float32]
    inputs = [-5, -2, -1, -0.5,  0, 0.5, 1, 2, 5]
    outputs = [-1, -1, -1, -1, 1, 1, 1, 1, 1]
    quantizers = [
        (Sign(), [0, 0, 1, 1, 1, 1, 1, 0, 0]),
        (Sign(STE()), [0, 0, 1, 1, 1, 1, 1, 0, 0]),
        (Sign(STE(1)), [0, 0, 1, 1, 1, 1, 1, 0, 0]),
        (Sign(STE(2)), [0, 1, 1, 1, 1, 1, 1, 1, 0]),
        (Sign(PolynomialSTE()), [0, 0, 0, 1, 2, 1, 0, 0, 0]),
        (Sign(SwishSTE()), nothing),
        (Sign(SwishSTE(5)), nothing),
        (Sign(SwishSTE(10)), nothing),
    ]

    # forward pass
    @testset "forward pass: $(q)" for (q, _) in quantizers
        @testset "input_type = $(T)" for T in Ts
            xt = T.(inputs)
            yt = T.(outputs)

            @testset "q($(x)) = $(y)" for (x, y) in zip(xt, yt)
                @test isa(q(x), T)
                @test q(x) == y
                @test isa(forward_pass(q, x), T)
                @test forward_pass(q, x) == y
            end
            @test isa(q(xt), typeof(xt))
            @test q(xt) == yt
            @test isa(forward_pass.(q, xt), typeof(xt))
            @test forward_pass.(q, xt) == yt
        end
    end

    # pullback
    @testset "pullback: $(q)" for (q, outputs_pullback) in quantizers
        @testset "input_type = $(T)" for T in Ts
            xt = T.(inputs)

            # no predefined outputs for Swish
            if isnothing(outputs_pullback)
                outputs_pullback = pullback.(q, xt)
            end
            yt = T.(outputs_pullback)

            @testset "pullback(q, $(x)) = $(y)" for (x, y) in zip(xt, yt)
                @test isa(pullback(q, x), T)
                @test pullback(q, x) ≈ y
            end

            @test isa(pullback.(q, xt), typeof(xt))
            @test pullback.(q, xt) ≈ yt
            @test gradient(x -> sum(q(x)), xt)[1] ≈ yt
        end
    end
end
