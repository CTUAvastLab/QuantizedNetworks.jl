"""
    ClippedArray(x::AbstractArray, lo::Real=-1, hi::Real=1)
    ClippedArray(dims...; lo::Real=-1, hi::Real=1, init = glorot_uniform)

Array type with elements clipped between `lo` and `hi`, defaults to [-1, +1].

# Examples

```jldoctest
julia> C = ClippedArray([-4.0, -0.8, 0.1, 1.2], -1, 1.2)
4-element ClippedArray{Float64, 1, Vector{Float64}}:
 -1.0
 -0.8
  0.1
  1.2

julia> C[[2, 4]] = [-2.4, 2.6]
2-element Vector{Float64}:
 -2.4
  2.6

julia> C
4-element ClippedArray{Float64, 1, Vector{Float64}}:
 -1.0
 -1.0
  0.1
  1.2

julia> using Random; Random.seed!(3);

julia> ClippedArray(2, 3)
2×3 ClippedArray{Float32, 2, Matrix{Float32}}:
  0.843751  -1.0  0.96547
 -0.351124   1.0  1.0
```
"""
mutable struct ClippedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    x::A
    lo::T
    hi::T

    function ClippedArray(
        x::AbstractArray{T,N},
        lo::Real=-1,
        hi::Real=1,
    ) where {T,N}

        lo, hi = T(lo), T(hi)
        return new{T,N,typeof(x)}(clamp.(x, lo, hi), lo, hi)
    end
end

function ClippedArray(
    dims::Union{Integer, AbstractUnitRange}...;
    lo::Real=-1,
    hi::Real=1,
    init = glorot_uniform,
)
    return ClippedArray(init(dims...), lo, hi)
end

const ClippedVector{T,A} = ClippedArray{T,1,A}
const ClippedMatrix{T,A} = ClippedArray{T,2,A}

Base.size(c::ClippedArray) = size(c.x)
Base.length(c::ClippedArray) = length(c.x)
Base.getindex(c::ClippedArray, args...) = getindex(c.x, args...)
Base.setindex!(c::ClippedArray, v, args...) = setindex!(c.x, clamp.(v, c.lo, c.hi), args...)
Base.zero(c::ClippedArray) = ClippedArray(zero(c.x), c.lo, c.hi)

Base.similar(c::ClippedArray) = ClippedArray(similar(c.x), c.lo, c.hi)
function Base.similar(c::ClippedArray, dims::Union{Integer, AbstractUnitRange}...)
    return ClippedArray(similar(c.x, dims...), c.lo, c.hi)
end

# ChainRulesCore utils
function ChainRulesCore.ProjectTo(c::ClippedArray)
    return ProjectTo{ClippedArray}(; parent = ProjectTo(c.x), lo = c.lo, hi = c.hi)
end
function (project::ProjectTo{ClippedArray})(dx::AbstractArray)
    return ClippedArray(project.parent(dx), project.lo, project.hi)
end
function (project::ProjectTo{ClippedArray})(dx::ClippedArray)
    return ClippedArray(project.parent(dx.x), project.lo, project.hi)
end
