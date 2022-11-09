@doc raw"""
    AbstractEstimator

Estimators are used for estimation of the gradient of quantizers for whic the true gradient does not exist. Estimators are used for dispatch on backward pass, i.e. for each quantizer (for example [`Sign`](@ref)) and estimator it is necessary to define specific method for [`pullback`](@ref) function.
"""
abstract type AbstractEstimator end

@doc raw"""
    STE(threshold::Real = 2)

For more details see [`AbstractEstimator`](@ref), [`Sign`](@ref), [`Heaviside`](@ref) or [`Ternary`](@ref).
"""
struct STE{T<:Real} <: AbstractEstimator
    threshold::T

    function STE(threshold::T=2) where {T<:Real}
        threshold > 0 || throw(ArgumentError("`threshold` must be positive"))
        return new{T}(threshold)
    end
end

Base.show(io::IO, e::STE) = print(io, "STE($(e.threshold))")

@doc raw"""
    PolynomialSTE()

For more details see [`AbstractEstimator`](@ref) or [`Sign`](@ref).
"""
struct PolynomialSTE <: AbstractEstimator end

Base.show(io::IO, ::PolynomialSTE) = print(io, "PolynomialSTE")


@doc raw"""
    SwishSTE(β:Real = 5)

For more details see [`AbstractEstimator`](@ref) or [`Sign`](@ref).
"""
struct SwishSTE{T} <: AbstractEstimator
    β::T

    function SwishSTE(β::T=5) where {T<:Real}
        β > 0 || throw(ArgumentError("`β` must be positive"))
        return new{T}(β)
    end
end

Base.show(io::IO, e::SwishSTE) = print(io, "SwishSTE($(e.β))")

@doc raw"""
    StochasticSTE()

For more details see [`AbstractEstimator`](@ref) or [`Sign`](@ref).
"""
struct StochasticSTE <: AbstractEstimator end

Base.show(io::IO, ::StochasticSTE) = print(io, "StochasticSTE")
