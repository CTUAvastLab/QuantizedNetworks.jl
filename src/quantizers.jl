abstract type Quantizer end

Base.broadcastable(q::Quantizer) = Ref(q)
(q::Quantizer)(x) = value.(q, x)

function ChainRulesCore.rrule(q::Quantizer, x)

    function quantizer_pullback(Δy)
        return NoTangent(), Δy .* deriv.(q, x)
    end
    return value.(q, x), quantizer_pullback
end

"""
    value(q::Quantizer, x::Real)

Applies quantizer `q` to value `x`.
"""
function value end

"""
    deriv(q::Quantizer, x::Real)

Returns gradient of `q` with respect to `x`
"""
function deriv end

"""
    Sign(threshold::Real = 1)

deterministic binary quantizer that return -1 when the given input is less than zero and 1 otherwise. The gradient is estimated using the Straight-Through Estimator (essentially the
binarization is replaced by a clipped identity on the backward pass).

# References

- [`Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1`](https://arxiv.org/abs/1602.02830)
"""
struct Sign{T} <: Quantizer
    threshold::T

    function Sign(threshold::T = 1) where {T<:Real}
        threshold > 0 || throw(ArgumentError("`threshold` must be positive"))
        return new{T}(threshold)
    end
end

value(::Sign, x::Real) = ifelse(x < 0, -one(x), one(x))
deriv(q::Sign, x::T) where {T<:Real} = T(abs(x) <= q.threshold)

"""
    PolySign(threshold::Real = 1)

TODO
"""
struct PolySign{T} <: Quantizer
    threshold::T

    function PolySign(threshold::T = 1) where {T<:Real}
        threshold > 0 || throw(ArgumentError("`threshold` must be positive"))
        return new{T}(threshold)
    end
end

value(::PolySign, x::Real) = ifelse(x < 0, -one(x), one(x))
deriv(q::PolySign, x::T) where {T<:Real} = T((2 - 2abs(x)) * (abs(x) <= q.threshold))

"""
    SwishSign(β)

TODO
"""
struct SwishSign{T} <: Quantizer
    β::T

    function SwishSign(β::T = 5) where {T<:Real}
        β > 0 || throw(ArgumentError("`β` must be positive"))
        return new{T}(β)
    end
end

value(::SwishSign, x::Real) = ifelse(x < 0, -one(x), one(x))
function deriv(q::SwishSign, x::T) where {T<:Real}
    β = q.β
    return T((β*(2 - β*x*tanh((β*x)/2)))/(1 + cosh(β * x)))
end

"""
    Ternary(Δ, threshold)

TODO
"""
struct Ternary{T} <: Quantizer
    Δ::T
    threshold::T

    function Ternary(Δ::Real = 0.005, threshold::Real = 1)
        Δ > 0 || throw(ArgumentError("`Δ` must be positive"))
        threshold > 0 || throw(ArgumentError("`threshold` must be positive"))

        Δ, threshold = promote(Δ, threshold)
        return new{typeof(Δ)}(Δ, threshold)
    end
end

function value(q::Ternary, x::Real)
    return if x < q.Δ
        -one(x)
    elseif x > q.Δ
        one(x)
    else
        zero(x)
    end
end
deriv(q::Ternary, x::T) where {T<:Real} = T(abs(x) <= q.threshold)
