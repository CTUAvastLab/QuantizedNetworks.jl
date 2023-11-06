@doc raw"""
    AbstractEstimator

Estimators are used for estimation of the gradient of quantizers for which the true gradient does not exist.

Estimators are used for dispatch on backward pass, i.e. for each quantizer (for example [`Sign`](@ref)) 
and it is necessary to define specific method for [`pullback`](@ref) function.
"""
abstract type AbstractEstimator end

@doc raw"""
    STE(threshold::Real = 2)

It is the simplest estimator used in all quantizers like [`Sign`](@ref), [`Heaviside`](@ref) or [`Ternary`](@ref).

It requires a real positive number for threshold parameter, in case a negative number is supplied an `ArgumentError` exception is thrown.

Threshold is used, in the `pullback` function, to determine the range of values for which the gradient is calculated by an estimation function. [`Sign`](@ref)
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

Currently is only supported for binary [`Sign`](@ref) quantizer. 

Does not require any additional parameters, simply indicates that a polynomial approximation of sign function is used and its respective derivative.
"""
struct PolynomialSTE <: AbstractEstimator end

Base.show(io::IO, ::PolynomialSTE) = print(io, "PolynomialSTE")


@doc raw"""
    SwishSTE(β:Real = 5)

Currently is only supported for binary [`Sign`](@ref) quantizer. 

It requires a real positive number for β parameter, in case a negative number is supplied an `ArgumentError` exception is thrown.

β is used as the parameter in the calculation of the swish function and its derivative in the `pullback` function. [`Sign`](@ref)
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

To be defined.
"""
struct StochasticSTE <: AbstractEstimator end

Base.show(io::IO, ::StochasticSTE) = print(io, "StochasticSTE")
