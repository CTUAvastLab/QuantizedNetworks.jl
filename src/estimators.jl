abstract type AbstractEstimator end

struct STE{T<:Real} <: AbstractEstimator
    threshold::T

    function STE(threshold::T = 2) where {T<:Real}
        threshold > 0 || throw(ArgumentError("`threshold` must be positive"))
        return new{T}(threshold)
    end
end

Base.show(io::IO, e::STE) = print(io, "STE($(e.threshold))")

struct PolynomialSTE <: AbstractEstimator end

Base.show(io::IO, ::PolynomialSTE) = print(io, "PolynomialSTE")

struct SwishSTE{T} <: AbstractEstimator
    β::T

    function SwishSTE(β::T = 5) where {T<:Real}
        β > 0 || throw(ArgumentError("`β` must be positive"))
        return new{T}(β)
    end
end

Base.show(io::IO, e::SwishSTE) = print(io, "SwishSTE($(e.β))")
