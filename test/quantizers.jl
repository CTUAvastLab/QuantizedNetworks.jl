using QuantizedNetworks: forward_pass, pullback

# Sign function
@testset "Sign quantizer" begin
    Ts = [Float64, Float32]
    inputs_real = [-5, -2, -1, -0.5,  0, 0.5, 1, 2, 5]
    inputs_missing = [-5, -2, -1, -0.5,  0, 0.5, 1, 2, 5]
    outputs = [-1, -1, -1, -1, 1, 1, 1, 1, 1]
    quantizers = [
        (Sign(), [0, 1, 1, 1, 1, 1, 1, 1, 0]),
        (Sign(STE()), [0, 1, 1, 1, 1, 1, 1, 1, 0]),
        (Sign(STE(1)), [0, 0, 1, 1, 1, 1, 1, 0, 0]),
        (Sign(STE(2)), [0, 1, 1, 1, 1, 1, 1, 1, 0]),
        (Sign(PolynomialSTE()), [0, 0, 0, 1, 2, 1, 0, 0, 0]),
        (Sign(SwishSTE()), nothing),
        (Sign(SwishSTE(5)), nothing),
        (Sign(SwishSTE(10)), nothing),
    ]

    # forward pass
    @testset "inputs = $(inputs)" for inputs in [inputs_real, inputs_missing]
        @testset "input_type = $(T)" for T in Ts
            @testset "Quantizer: $(q)" for (q, outputs_pullback) in quantizers
                contains_missing = any(ismissing, inputs)

                # inputs/outputs conversion
                if contains_missing
                    xt = convert(Vector{Union{T, Missing}}, inputs)
                else
                    xt = T.(inputs)
                end
                yt = T.(outputs)

                # no predefined outputs for Swish
                Δyt = isnothing(outputs_pullback) ? pullback(q, xt) : outputs_pullback

                # forward pass
                @testset "forward pass" begin
                    @testset "q($(x)) = $(y)" for (x, y) in zip(xt, yt)
                        contains_missing || @test isa(q(x), T)
                        @test q(x) == y
                        contains_missing || @test isa(forward_pass(q, x), T)
                        @test forward_pass(q, x) == y
                    end
                    @test isa(q(xt), Vector{T})
                    @test q(xt) == yt
                    @test isa(forward_pass(q, xt), Vector{T})
                    @test forward_pass(q, xt) == yt
                end

                # pullback
                @testset "pullback" begin
                    @testset "pullback(q, $(x)) = $(Δy)" for (x, Δy) in zip(xt, Δyt)
                        contains_missing || @test isa(pullback(q, x), T)
                        @test pullback(q, x) ≈ Δy
                    end

                    @test isa(pullback.(q, xt), Vector{T})
                    @test pullback(q, xt) ≈ Δyt
                    @test gradient(x -> sum(q(x)), xt)[1] ≈ Δyt
                end
            end
        end
    end
end
