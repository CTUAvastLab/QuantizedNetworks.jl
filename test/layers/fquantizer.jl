using QuantizedNetworks: forward_pass

inputs = [
    rand(7, 3),
    hcat(randn(2, 3), [1,missing]),
]

ks = 1:2:10

@testset "FQuantizer" begin
    @testset "input = $(input)" for input in inputs
        @testset "T = $(T)" for T in [Float32]
            contains_missing = any(ismissing, input)
            if contains_missing
                continue
            end
            if contains_missing
                x = convert(Matrix{Union{T, Missing}}, input)
            else
                x = T.(input)
            end

            @testset "k = $k" for k in ks
                q = FQuantizer((size(x, 1), k))
                w, b = q.weight, q.bias

                @testset "forward pass" begin
                    y = x .* reshape(w, :, 1, k) .+ reshape(b, :, 1, k)
                    y = hcat([vec(slc') for slc in eachslice(y; dims = 2)]...)
                    y = ifelse.(
                        ismissing.(y),
                        -one(T),
                        ifelse.(y .<= 0, -one(T), one(T))
                    )

                    @test isa(q(x), Matrix{T})
                    @test q(x) == y
                end

                if !contains_missing
                    @testset "pullback" begin
                        _fw(w) = sum(forward_pass(w, b, x))
                        _fb(b) = sum(forward_pass(w, b, x))
                        _fx(x) = sum(forward_pass(w, b, x))

                        ps = Flux.params([q, x])
                        gs = Flux.gradient(() -> sum(forward_pass(w, b, x)), ps)

                        gs[ps[1]] ≈ grad(central_fdm(5, 1), _fw, w)[1]
                        gs[ps[2]] ≈ grad(central_fdm(5, 1), _fb, b)[1]
                        gs[ps[3]] ≈ grad(central_fdm(5, 1), _fx, x)[1]

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
