abstract type AbstractQuantizer{E <: AbstractEstimator} end

Base.broadcastable(q::AbstractQuantizer) = Ref(q)
NNlib.fast_act(q::AbstractQuantizer, ::AbstractArray) = q

(q::AbstractQuantizer)(x) = forward_pass(q, x)

"""
    forward_pass(q::Quantizer, x)

Applies quantizer `q` to value `x`.
"""
function forward_pass(q::AbstractQuantizer, x)
    T = nonmissingtype(eltype(x))
    return T.(forward_pass.(q, x))
end

"""
    pullback(q::Quantizer, x)

Returns gradient of `q` with respect to `x`.
"""
function pullback(q::AbstractQuantizer, x)
    T = nonmissingtype(eltype(x))
    return T.(pullback.(q, x))
end

function ChainRulesCore.rrule(q::AbstractQuantizer, x)
    y = q(x)
    project_y = ProjectTo(y)

    function quantizer_pullback(Δy)
        return NoTangent(), project_y(Δy .* pullback(q, x))
    end
    return y, quantizer_pullback
end

"""
    Sign(estimator)

deterministic binary quantizer that return -1 when the given input is less than zero and 1 otherwise.

# References

- [`Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1`](https://arxiv.org/abs/1602.02830)
"""
struct Sign{E<:AbstractEstimator} <: AbstractQuantizer{E}
    estimator::E

    function Sign(estimator::E = STE()) where {E}
        return new{E}(estimator)
    end
end

Base.show(io::IO, q::Sign) = print(io, "Sign($(q.estimator))")

forward_pass(::Sign, x::Missing) = -1
forward_pass(::Sign, x::Real) = ifelse(x < 0, -one(x), one(x))

pullback(q::Sign, x::Missing) = 0
function pullback(q::Sign{<:STE}, x::T)::T where {T<:Real}
    t = q.estimator.threshold
    return abs(x) <= t
end

function pullback(::Sign{<:PolynomialSTE}, x::T)::T where {T<:Real}
    return abs(2 - 2abs(x)) * (abs(x) <= 1)
end

function pullback(q::Sign{<:SwishSTE}, x::T)::T where {T<:Real}
    β = q.estimator.β
    return (β*(2 - β*x*tanh((β*x)/2)))/(1 + cosh(β * x))
end

function ChainRulesCore.rrule(q::Sign{<:StochasticSTE}, x)
    T = nonmissingtype(eltype(x))
    y = q(x .+ (2 .* rand(T, size(x)) .- 1))

    return y, Δy -> (NoTangent(), Δy)
end

"""
    Heaviside(estimtor)

TODO
"""
struct Heaviside{E<:AbstractEstimator} <: AbstractQuantizer{E}
    estimator::E

    function Heaviside(estimator::E = STE()) where {E}
        return new{E}(estimator)
    end
end

Base.show(io::IO, q::Heaviside) = print(io, "Heaviside($(q.estimator))")
forward_pass(::Heaviside, x::Missing) = 0
forward_pass(::Heaviside, x::Real) = ifelse(x <= 0, zero(x), one(x))

pullback(q::Heaviside, x::Missing) = 0
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

    function Ternary(Δ::T = 0.05, estimator::E = STE()) where {T, E}
        Δ > 0 || throw(ArgumentError("`Δ` must be positive"))
        return new{E, T}(Δ, estimator)
    end
end

Base.show(io::IO, q::Ternary) = print(io, "Ternary$((q.Δ, q.estimator))")

forward_pass(::Ternary, x::Missing) = 0
function forward_pass(q::Ternary, x::Real)
    return if x < -q.Δ
        -one(x)
    elseif x > q.Δ
        one(x)
    else
        zero(x)
    end
end

pullback(q::Ternary, x::Missing) = 0
function pullback(q::Ternary{<:STE}, x::T)::T where {T<:Real}
    t = q.estimator.threshold
    return abs(x) <= t
end
