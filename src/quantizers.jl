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
    Sign(lo, hi, threshold)

deterministic binary quantizer that return `lo` when the given input is less than zero and `hi` otherwise.

# TODO:
- better description of input arguments and rrule implementation

# References

- [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830)

# Examples
```julia
julia> q = Sign()
Sign{Int64}(-1, 1, 1)

julia> value.(q, [-2, 0.5, 2])
3-element Vector{Float64}:
 -1.0
  1.0
  1.0
```
"""
struct Sign{T} <: Quantizer
    lo::T
    hi::T
    threshold::T

    Sign(lo::Real, hi::Real, threshold::Real) = Sign(promote(lo, hi, threshold)...)

    function Sign(lo::T, hi::T, threshold::T) where {T<:Real}
        lo < hi || throw(ArgumentError("`lo` must be less than `hi`"))
        threshold > 0 || throw(ArgumentError("`threshold` must be positive"))
        return new{T}(lo, hi, threshold)
    end
end

Sign(; lo::Real=-1, hi::Real=1, threshold::Real=1) = Sign(lo, hi, threshold)

value(q::Sign, x::T) where {T<:Real} = ifelse(x < 0, T(q.lo), T(q.hi))
deriv(q::Sign, x::T) where {T<:Real} = T(abs(x) <= q.threshold)



