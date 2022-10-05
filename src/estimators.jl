abstract type AbstractEstimator end

struct STE{T<:Real} <: AbstractEstimator
    threshold::T

    function STE(threshold::T = 1) where {T<:Real}
        threshold > 0 || throw(ArgumentError("`threshold` must be positive"))
        return new{T}(threshold)
    end
end

struct PolynomialSTE{T} <: AbstractEstimator
    threshold::T

    function PolynomialSTE(threshold::T = 1) where {T<:Real}
        threshold > 0 || throw(ArgumentError("`threshold` must be positive"))
        return new{T}(threshold)
    end
end

struct SwishSTE{T} <: AbstractEstimator
    β::T

    function SwishSTE(β::T = 5) where {T<:Real}
        β > 0 || throw(ArgumentError("`β` must be positive"))
        return new{T}(β)
    end
end
