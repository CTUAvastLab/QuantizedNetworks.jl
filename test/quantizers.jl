using QuantizedNetworks: forward_pass, pullback

Ts = [Float16, Float32, Float64]
inputs_real = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]
inputs_missing = [-5, -2, -1, -0.5, missing, 0.5, 1, 2, missing]

test_values = [
(
    (Sign(), Sign(STE()), Sign(STE(2))),
    (
        (inputs_real, [-1, -1, -1, -1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0]),
        (inputs_missing, [-1, -1, -1, -1, -1, 1, 1, 1, -1], [0, 1, 1, 1, 0, 1, 1, 1, 0]),
    ),
),
(
    (Sign(STE()), ),
    (
        (inputs_real, [-1, -1, -1, -1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0]),
        (inputs_missing, [-1, -1, -1, -1, -1, 1, 1, 1, -1], [0, 1, 1, 1, 0, 1, 1, 1, 0]),
    ),
),
(
    (Sign(PolynomialSTE()),),
    (
        (inputs_real, [-1, -1, -1, -1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 2, 1, 0, 0, 0]),
        (inputs_missing, [-1, -1, -1, -1, -1, 1, 1, 1, -1], [0, 0, 0, 1, 0, 1, 0, 0, 0]),
    ),
),
(
    (Sign(SwishSTE()), Sign(SwishSTE(5)), Sign(SwishSTE(10))),
    (
        (inputs_real, [-1, -1, -1, -1, 1, 1, 1, 1, 1], nothing),
        (inputs_missing, [-1, -1, -1, -1, -1, 1, 1, 1, -1], nothing),
    ),
),
(
    (Heaviside(), Heaviside(STE()), Heaviside(STE(2))),
    (
        (inputs_real, [0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0]),
        (inputs_missing, [0, 0, 0, 0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 0, 1, 1, 1, 0]),
    ),
),
(
    (Heaviside(STE(1)),),
    (
        (inputs_real, [0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 0, 0]),
        (inputs_missing, [0, 0, 0, 0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1, 0, 0]),
    ),
),
(
    (Ternary(), Ternary(0.05, STE(2)), Ternary(0.05, STE(2))),
    (
        (inputs_real, [-1, -1, -1, -1, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0]),
        (inputs_missing, [-1, -1, -1, -1, 0, 1, 1, 1, 0], [0, 1, 1, 1, 0, 1, 1, 1, 0]),
    ),
),
(
    (Ternary(0.05, STE(1)),),
    (
        (inputs_real, [-1, -1, -1, -1, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 0, 0]),
        (inputs_missing, [-1, -1, -1, -1, 0, 1, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1, 0, 0]),
    ),
),
(
    (Ternary(0.6), Ternary(0.6, STE()), Ternary(0.6, STE(2))),
    (
        (inputs_real, [-1, -1, -1, 0, 0, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 0]),
        (inputs_missing, [-1, -1, -1, 0, 0, 0, 1, 1, 0], [0, 1, 1, 1, 0, 1, 1, 1, 0]),
    ),
),
(
    (Ternary(0.6, STE(1)),),
    (
        (inputs_real, [-1, -1, -1, 0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 0, 0]),
        (inputs_missing, [-1, -1, -1, 0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1, 0, 0]),
    ),
),
]

# Quantizers
for (quantizers, inouts) in test_values
    # forward pass
    @testset "Quantizer: $(q)" for q in quantizers
        @testset "input_type = $(T)" for T in Ts
            @testset "inputs = $(inputs)" for (inputs, outputs, outputs_pullback) in inouts
                contains_missing = any(ismissing, inputs)

                # inputs/outputs conversion
                if contains_missing
                    xt = convert(Vector{Union{T,Missing}}, inputs)
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

                    @test isa(pullback(q, xt), Vector{T})
                    @test pullback(q, xt) ≈ Δyt
                    @test gradient(x -> sum(q(x)), xt)[1] ≈ Δyt
                end
            end
        end
    end
end
