v = collect(-5:0.5:5)
lims = [
    (-1, 1),
    (-2, 2),
    (2, 3),
    (-3, -2),
]

@testset "A is $(typeof(A))" for A in [v, reshape(copy(v), 7, 3)]
    @testset "(lo, hi) = ($(lo), $(hi))" for (lo, hi) in lims
        Ac = ClippedArray(A, lo, hi)

        @test size(Ac) == size(A)
        @test length(Ac) == length(A)
        @test zero(Ac).x == clamp.(zero(A), lo, hi)
        @test extrema(Ac) == (lo, hi)

        Ac[1] = lo - 1
        Ac[end] = hi + 1
        @test Ac[1] == lo
        @test Ac[end] == hi

        project = ProjectTo(Ac)
        @test project(A) == Ac
        @test isa(project(A), ClippedArray)
        @test project(Ac) == Ac
        @test isa(project(Ac), ClippedArray)
    end
end
