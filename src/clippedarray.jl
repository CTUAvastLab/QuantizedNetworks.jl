"""
    ClippedArray

Array with elements clipped between `lo` and `hi`, defaults to [-1, +1].
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

const ClippedVector{T,A} = ClippedArray{T,1,A}
const ClippedMatrix{T,A} = ClippedArray{T,2,A}

Base.size(c::ClippedArray) = size(c.x)
Base.length(c::ClippedArray) = length(c.x)
Base.getindex(c::ClippedArray, args...) = getindex(c.x, args...)
Base.setindex!(c::ClippedArray, v, args...) = setindex!(c.x, clamp.(v, c.lo, c.hi), args...)
Base.zero(c::ClippedArray) = Base.zero(c::ClippedArray) = ClippedArray(zero(c.x), c.lo, c.hi)


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
