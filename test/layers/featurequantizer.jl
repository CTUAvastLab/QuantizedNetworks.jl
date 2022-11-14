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

                if !contains_missing
                    @testset "pullback" begin
                        _fw(w) = sum(_forward_pass(w, b, x))
                        _fb(b) = sum(_forward_pass(w, b, x))
                        _fx(x) = sum(_forward_pass(w, b, x))

                        ps = Flux.params([q, x])
                        gs = Flux.gradient(() -> sum(_forward_pass(w, b, x)), ps)

                        gs[ps[1]] ≈ grad(central_fdm(5, 1), _fw, w)[1]
                        gs[ps[2]] ≈ grad(central_fdm(5, 1), _fb, b)[1]
                        gs[ps[3]] ≈ grad(central_fdm(5, 1), _fx, x)[1]

                        if output_quantizer == identity
                            ps = Flux.params([q, x])
                            gs = Flux.gradient(() -> sum(q(x)), ps)

                            gs[ps[1]] ≈ grad(central_fdm(5, 1), _fw, w)[1]
                            gs[ps[2]] ≈ grad(central_fdm(5, 1), _fb, b)[1]
                            gs[ps[3]] ≈ grad(central_fdm(5, 1), _fx, x)[1]
                        end
                    end
                end
            end
        end
    end
end
