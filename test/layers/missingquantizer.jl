Ts = [Float16, Float32, Float64]
input = [1, 2, missing, -1, missing, -2]
test_quanizers = [
    (Sign(), [1, 1, -1, 1, -1, 1]),
    (Sign(STE(1)), [1, 1, -1, 1, -1, 1]),
    (Sign(STE(2)), [1, 1, -1, 1, -1, 1]),
    (Sign(PolynomialSTE()), [1, 1, -1, 1, -1, 1]),
    (Sign(SwishSTE()), [1, 1, -1, 1, -1, 1]),
    (Sign(SwishSTE(5)), [1, 1, -1, 1, -1, 1]),
    (Heaviside(), [1, 1, 0, 1, 0, 1]),
    (Heaviside(STE(1)), [1, 1, 0, 1, 0, 1]),
    (Heaviside(STE(2)), [1, 1, 0, 1, 0, 1]),
    (Ternary(), [1, 1, -1, 1, -1, 1]),
    (Ternary(0.05, STE(1)), [1, 1, -1, 1, -1, 1]),
    (Ternary(0.05, STE(2)), [1, 1, -1, 1, -1, 1]),
]

@testset "Quantizer: $(q)" for (q, output) in test_quanizers
    @testset "input_type = $(T)" for T in Ts
        contains_missing = any(ismissing, input)

        # inputs/outputs conversion
        if contains_missing
            xt = convert(Vector{Union{T,Missing}}, input)
        else
            xt = T.(input)
        end
        yt = T.(output)

        f1 = MissingQuantizer(q)
        f2 = MissingQuantizer(; quantizer = q)

        @test isa(f1(xt), Vector{T})
        @test f1(xt) == yt

        @test isa(f2(xt), Vector{T})
        @test f2(xt) == yt
    end
end
