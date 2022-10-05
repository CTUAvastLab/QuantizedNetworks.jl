abstract type AbstractQuantizer{E <: AbstractEstimator} end

Base.broadcastable(q::AbstractQuantizer) = Ref(q)
(q::AbstractQuantizer)(x) = forward_pass.(q, x)

function ChainRulesCore.rrule(q::AbstractQuantizer, x)

    function quantizer_pullback(Δy)
        return NoTangent(), Δy .* pullback.(q, x)
    end
    return forward_pass.(q, x), quantizer_pullback
end

"""
    forward_pass(q::Quantizer, x::Real)

Applies quantizer `q` to value `x`.
"""
function forward_pass end

"""
    pullback(q::Quantizer, x::Real)

Returns gradient of `q` with respect to `x`
"""
function pullback end

"""
    Sign(estimator)

deterministic binary quantizer that return -1 when the given input is less than zero and 1 otherwise.

# References

- [`Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1`](https://arxiv.org/abs/1602.02830)
"""
struct Sign{E<:AbstractEstimator} <: AbstractQuantizer{E}
    estimator::E

    function Sign(estimator::E = STE(1)) where {E}
        return new{E}(estimator)
    end
end

Base.show(io::IO, q::Sign) = print(io, "Sign($(q.estimator))")
forward_pass(::Sign, x::Real) = ifelse(x < 0, -one(x), one(x))

function pullback(q::Sign{<:STE}, x::T)::T where {T<:Real}
    t = q.estimator.threshold
    return abs(x) <= t
end

function pullback(q::Sign{<:PolynomialSTE}, x::T)::T where {T<:Real}
    t = q.estimator.threshold
    return (2 - 2abs(x)) * (abs(x) <= t)
end

function pullback(q::Sign{<:SwishSTE}, x::T)::T where {T<:Real}
    β = q.estimator.β
    return (β*(2 - β*x*tanh((β*x)/2)))/(1 + cosh(β * x))
end

"""
    Heaviside(estimtor)

TODO
"""
struct Heaviside{E<:AbstractEstimator} <: AbstractQuantizer{E}
    estimator::E

    function Heaviside(estimator::E = STE(1)) where {E}
        return new{E}(estimator)
    end
end

Base.show(io::IO, q::Heaviside) = print(io, "Heaviside($(q.estimator))")
forward_pass(::Heaviside, x::Real) = ifelse(x < 0, zero(x), one(x))

function pullback(q::Heaviside{<:STE}, x::T)::T where {T<:Real}
    t = q.estimator.threshold
    return abs(x) <= t
end

"""
    Ternary(Δ, estimator)

TODO
"""
struct Ternary{E<:AbstractEstimator, T} <: AbstractQuantizer{E}
    Δ::T
    estimator::E

    function Ternary(Δ::T = 0.005, estimator::E = STE(1)) where {T, E}
        Δ > 0 || throw(ArgumentError("`Δ` must be positive"))
        return new{E, T}(Δ, estimator)
    end
end

Base.show(io::IO, q::Ternary) = print(io, "Ternary$((q.Δ, q.estimator))")

function forward_pass(q::Ternary, x::Real)
    return if x < q.Δ
        -one(x)
    elseif x > q.Δ
        one(x)
    else
        zero(x)
    end
end

function pullback(q::Ternary{<:STE}, x::T)::T where {T<:Real}
    t = q.estimator.threshold
    return abs(x) <= t
end
