using QuantizedNetworks: _forward_pass

output_quantizers = [
    identity,
    Sign(),
    Sign(STE()),
    Sign(STE(1)),
    Sign(STE(2)),
    Sign(PolynomialSTE()),
    Sign(SwishSTE()),
    Sign(SwishSTE(5)),
    Sign(SwishSTE(10)),
]

inputs = [
    rand(7, 3),
    hcat(randn(2, 3), [1,missing]),
]

ks = 1:2:10

@testset "FeatureQuantizer" begin
    @testset "quantizer = $(output_quantizer)" for output_quantizer in output_quantizers
        @testset "input = $(input)" for input in inputs
            @testset "T = $(T)" for T in [Float32, Float64]
                contains_missing = any(ismissing, input)
                if contains_missing && (output_quantizer == identity)
                    continue
                end
                if contains_missing
                    x = convert(Matrix{Union{T, Missing}}, input)
                else
                    x = T.(input)
                end

                @testset "k = $k" for k in ks
                    q = FeatureQuantizer(size(x, 1), k; output_quantizer)
                    w, b = q.weight, q.bias

                    @testset "forward pass" begin
                        y = x .* reshape(w, :, 1, k) .+ reshape(b, :, 1, k)
                        y = hcat([vec(slc') for slc in eachslice(y; dims = 2)]...)

                        @test isa(q(x), Matrix{T})
                        if !contains_missing
                            @test _forward_pass(q.weight, q.bias, x) == y
                        end
                        @test q(x) == output_quantizer(y)
                    end

                    @testset "pullback" begin

                    end
                end
            end
        end
    end
end
