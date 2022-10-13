# tests only for constructor - other test are in quantizers.jl
@testset "constructors: $(T)" for T in [STE, SwishSTE]
    @test_throws ArgumentError T(0)
    @test_throws ArgumentError T(-1)
end
